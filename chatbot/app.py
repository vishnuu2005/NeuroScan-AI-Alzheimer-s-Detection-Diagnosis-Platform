import time
import re
import os
import json
import uuid

# Ensure Matplotlib uses a writable local cache in all environments
if 'MPLCONFIGDIR' not in os.environ:
    _mpl_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.matplotlib_cache')
    os.makedirs(_mpl_dir, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = _mpl_dir

from datetime import datetime
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, send_file, flash
from PIL import Image
import torch
import torchvision.transforms as transforms
from dotenv import load_dotenv
from groq import Groq
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy

try:
    from .inference import load_model, predict, transform_image
    from .generate_reports import generate_report
    from .validators import validate_mri_image, allowed_file, safe_remove, ALLOWED_EXTENSIONS
    from .formatters import get_risk_tier, format_stage_label, prepare_chart_dataset
    from .clinical_ai import CLINICAL_SYSTEM_PROMPT, generate_local_clinical_response
    from .logging_config import configure_logging, log_clinical_event
except ImportError:
    from inference import load_model, predict, transform_image
    from generate_reports import generate_report
    from validators import validate_mri_image, allowed_file, safe_remove, ALLOWED_EXTENSIONS
    from formatters import get_risk_tier, format_stage_label, prepare_chart_dataset
    from clinical_ai import CLINICAL_SYSTEM_PROMPT, generate_local_clinical_response
    from logging_config import configure_logging, log_clinical_event

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
USE_GROQ = os.getenv("USE_GROQ", "1").lower() in ("1", "true", "yes")
client = None
if USE_GROQ:
    if not groq_api_key:
        print("Warning: GROQ_API_KEY not found in environment variables. Groq disabled.")
    else:
        try:
            client = Groq(api_key=groq_api_key)
        except Exception as e:
            # do not crash app on invalid key; fall back to local responses
            print("Groq client initialization failed:", repr(e))
            client = None

# ==============================
# Flask + DB Setup
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
UPLOADS_DIR = os.path.join(STATIC_DIR, 'uploads')
REPORTS_DIR = os.path.join(STATIC_DIR, 'reports')

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

app = Flask(__name__, static_folder=STATIC_DIR, template_folder=TEMPLATES_DIR)
configure_logging(app)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "neuroscan_secret_2024")
app.config['SESSION_COOKIE_NAME'] = 'neuroscan_session'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = os.getenv('SESSION_COOKIE_SAMESITE', 'Lax')
app.config['SESSION_COOKIE_SECURE'] = os.getenv('SESSION_COOKIE_SECURE', '0').lower() in ('1', 'true', 'yes')
app.config['PERMANENT_SESSION_LIFETIME'] = int(os.getenv('PERMANENT_SESSION_LIFETIME', '86400'))
app.config['SESSION_REFRESH_EACH_REQUEST'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = int(os.getenv('SEND_FILE_MAX_AGE_DEFAULT', '31536000'))
app.config['UPLOAD_FOLDER'] = UPLOADS_DIR
app.config['REPORTS_FOLDER'] = REPORTS_DIR
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit
# Database Configuration - Robust path resolution
custom_db_uri = os.getenv("SQLALCHEMY_DATABASE_URI")
if custom_db_uri:
    app.config['SQLALCHEMY_DATABASE_URI'] = custom_db_uri
else:
    local_inst = os.path.join(BASE_DIR, 'instance')
    root_inst = os.path.join(os.path.dirname(BASE_DIR), 'instance')
    
    db_file = None
    for cand in [os.path.join(local_inst, 'neuroscan.db'), os.path.join(root_inst, 'neuroscan.db')]:
        if os.path.exists(cand) and os.path.getsize(cand) > 0:
            db_file = cand
            break
            
    if not db_file:
        os.makedirs(local_inst, exist_ok=True)
        db_file = os.path.join(local_inst, 'neuroscan.db')
        
    db_uri_path = os.path.abspath(db_file).replace('\\', '/')
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_uri_path}'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

def init_db(application=app):
    """Initialize database tables and run automatic migrations within application context."""
    with application.app_context():
        db.create_all()
        # Light auto-migration for newly introduced metadata columns
        try:
            with db.engine.connect() as conn:
                cols = [row[1] for row in conn.execute(db.text("PRAGMA table_info(scan)")).fetchall()]
                if cols and 'processing_ms' not in cols:
                    conn.execute(db.text("ALTER TABLE scan ADD COLUMN processing_ms FLOAT DEFAULT 0.0"))
                if cols and 'model_version' not in cols:
                    conn.execute(db.text("ALTER TABLE scan ADD COLUMN model_version VARCHAR(50) DEFAULT 'ResNet18-Adaptive'"))
                # Composite performance indexes
                conn.execute(db.text("CREATE INDEX IF NOT EXISTS idx_scan_user_created ON scan (user_id, created_at)"))
                conn.execute(db.text("CREATE INDEX IF NOT EXISTS idx_chat_session_user_created ON chat_session (user_id, created_at)"))
                conn.execute(db.text("CREATE INDEX IF NOT EXISTS idx_chat_message_session_created ON chat_message (chat_session_id, created_at)"))
                conn.commit()
        except Exception as e:
            print("Auto-migration notice:", e)

# Ensure database tables exist automatically on application import
init_db(app)

@app.cli.command('init-db')
def init_db_command():
    """CLI command to create database tables."""
    init_db(app)
    print("Initialized database tables.")

# ==============================
# Database Models
# ==============================
class User(db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    name          = db.Column(db.String(100), nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password      = db.Column(db.String(200), nullable=False)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)
    scans         = db.relationship('Scan', backref='user', lazy=True, cascade='all, delete-orphan')
    chat_sessions = db.relationship('ChatSession', backref='user', lazy=True, cascade='all, delete-orphan')

class Scan(db.Model):
    __table_args__ = (
        db.Index('idx_scan_user_created', 'user_id', 'created_at'),
    )
    id            = db.Column(db.Integer, primary_key=True)
    user_id       = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    prediction    = db.Column(db.String(50), nullable=False)
    confidence    = db.Column(db.Float, nullable=False)
    all_probs     = db.Column(db.Text, nullable=False)
    warning       = db.Column(db.String(200))
    processing_ms = db.Column(db.Float, default=0.0)
    model_version = db.Column(db.String(50), default='ResNet18-Adaptive')
    created_at    = db.Column(db.DateTime, default=datetime.utcnow, index=True)

class ChatSession(db.Model):
    __table_args__ = (
        db.Index('idx_chat_session_user_created', 'user_id', 'created_at'),
    )
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    scan_id    = db.Column(db.Integer, db.ForeignKey('scan.id'), nullable=True)
    scan_stage = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    messages   = db.relationship('ChatMessage', backref='chat_session', lazy=True, cascade='all, delete-orphan')

    @property
    def message_count(self):
        return len(self.messages)

class ChatMessage(db.Model):
    __table_args__ = (
        db.Index('idx_chat_message_session_created', 'chat_session_id', 'created_at'),
    )
    id              = db.Column(db.Integer, primary_key=True)
    chat_session_id = db.Column(db.Integer, db.ForeignKey('chat_session.id'), nullable=False, index=True)
    role            = db.Column(db.String(20), nullable=False)
    content         = db.Column(db.Text, nullable=False)
    created_at      = db.Column(db.DateTime, default=datetime.utcnow)

# ==============================
# Auth Helpers
# ==============================
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to continue.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def current_user():
    if 'user_id' in session:
        return User.query.get(session['user_id'])
    return None

# ==============================
# Model Setup
# ==============================
classes = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
# Resolve model path robustly: prefer configured MODEL_PATH or best checkpoint
custom_model_env = os.getenv("MODEL_PATH")
possible_model_paths = []
if custom_model_env:
    possible_model_paths.append(custom_model_env)

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)
candidate_names = ["resnet2d_alzheimers_best.pth", "resnet2d_alzheimers.pth"]
candidate_dirs = [
    os.path.join(project_root, "models"),
    os.path.join(os.getcwd(), "models"),
    os.path.join(os.getcwd(), "AlzeigmersChatBot", "models"),
    os.path.join(base_dir, "models"),
    "models"
]
for d in candidate_dirs:
    for name in candidate_names:
        possible_model_paths.append(os.path.join(d, name))

model_path = None
alz_model = None
for p in possible_model_paths:
    p_norm = os.path.normpath(p)
    if os.path.exists(p_norm):
        try:
            alz_model = load_model(p_norm, num_classes=len(classes))
            model_path = p_norm
            break
        except Exception as e:
            continue

if alz_model is None:
    print("Warning: No compatible ResNet checkpoint loaded. Checked: " + ", ".join(possible_model_paths))

# REMOVED: local transform_image definition (was overwriting the import)
# REMOVED: IMAGE_SIZE constant (defined inside inference.py)

stage_info = {
    "NonDemented":              "No signs of dementia detected in this scan.",
    "VeryMildDemented":         "Very early stage. Slight memory lapses may be present. Early intervention is beneficial.",
    "MildDemented":             "Mild cognitive decline detected. Daily tasks are mostly manageable but support may help.",
    "ModerateDemented":         "Moderate decline detected. Daily assistance is likely needed. Please consult a specialist.",
    "Uncertain":                "The model could not confidently classify this scan.",
    "Unknown / Low Confidence": "The model could not confidently classify this scan."
}
stage_risk = {
    "NonDemented":              "low",
    "VeryMildDemented":         "early",
    "MildDemented":             "mild",
    "ModerateDemented":         "moderate",
    "Uncertain":                "unknown",
    "Unknown / Low Confidence": "unknown"
}

# ==============================
# Chat Helpers
# ==============================
def get_or_create_chat_session():
    user = current_user()
    if not user:
        return None
    session_id = session.get('chat_session_id')
    curr_scan_id = session.get('scan_db_id')
    curr_stage   = session.get('stage')

    if session_id:
        chat_sess = ChatSession.query.get(session_id)
        if chat_sess and chat_sess.user_id == user.id:
            # Sync session scan context if a new scan was processed
            if curr_scan_id and chat_sess.scan_id != curr_scan_id:
                chat_sess.scan_id = curr_scan_id
                chat_sess.scan_stage = curr_stage
                db.session.commit()
            return chat_sess

    chat_sess  = ChatSession(user_id=user.id, scan_id=curr_scan_id, scan_stage=curr_stage)
    db.session.add(chat_sess)
    db.session.commit()
    session['chat_session_id'] = chat_sess.id
    return chat_sess

def generate_stage_context():
    user = current_user()
    stage      = session.get('stage')
    confidence = session.get('confidence')

    if not stage and user:
        latest_scan = Scan.query.filter_by(user_id=user.id).order_by(Scan.created_at.desc()).first()
        if latest_scan:
            stage = latest_scan.prediction
            confidence = latest_scan.confidence
            session['stage'] = stage
            session['confidence'] = confidence
            session['scan_db_id'] = latest_scan.id

    if stage and confidence is not None:
        advice = {
            "NonDemented":      "The MRI suggests no signs of dementia. Provide reassurance and preventive lifestyle guidance.",
            "VeryMildDemented": "Early signs detected. Discuss early symptoms, monitoring, and lifestyle changes.",
            "MildDemented":     "Mild cognitive decline. Provide structured guidance and caregiver planning advice.",
            "ModerateDemented": "Moderate progression. Focus on caregiving strategies and medical supervision.",
            "Uncertain":        "The model could not confidently classify the scan. Advise professional evaluation."
        }.get(stage, "")
        return f"The MRI prediction stage is {stage} with {confidence:.2f}% confidence. {advice}"
    return "No MRI scan has been uploaded yet. Provide general neuro-health information and recommend uploading a scan."

def generate_chatbot_response(user_message):
    stage_context = generate_stage_context()
    chat_sess     = get_or_create_chat_session()
    current_stage = session.get('stage', 'NonDemented')
    current_conf  = session.get('confidence', 90.0)
    current_probs = session.get('all_probs')

    history = []
    if chat_sess:
        msgs = ChatMessage.query.filter_by(chat_session_id=chat_sess.id)\
                                .order_by(ChatMessage.created_at.desc())\
                                .limit(10).all()
        for m in reversed(msgs):
            history.append({"role": m.role, "content": m.content})

    reply = None
    if client and USE_GROQ:
        messages = [
            {"role": "system", "content": CLINICAL_SYSTEM_PROMPT},
            {"role": "system", "content": stage_context},
            *history,
            {"role": "user", "content": user_message}
        ]
        try:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.5,
                max_tokens=1024
            )
            reply = completion.choices[0].message.content.strip()
        except Exception as e:
            err_str = str(e).lower()
            if "rate_limit" in err_str or "429" in err_str:
                print("Notice: Groq API rate limit reached, falling back to local clinical knowledge engine.")
            elif "auth" in err_str or "401" in err_str or "key" in err_str:
                print("Notice: Groq API key unauthorized or unconfigured, utilizing local clinical knowledge engine.")
            else:
                print(f"Notice: Groq service communication issue ({e}), switching to local clinical knowledge engine.")
            reply = None

    if not reply:
        # High quality offline clinical knowledge fallback
        reply = generate_local_clinical_response(
            user_message=user_message,
            stage=current_stage,
            confidence=current_conf,
            all_probs=current_probs
        )

    if chat_sess:
        try:
            db.session.add(ChatMessage(chat_session_id=chat_sess.id, role='user',      content=user_message))
            db.session.add(ChatMessage(chat_session_id=chat_sess.id, role='assistant', content=reply))
            db.session.commit()
        except Exception as db_err:
            db.session.rollback()
            print("Failed to record chat history:", db_err)

    return reply

# ==============================
# Routes — Auth
# ==============================
DUMMY_PASSWORD_HASH = generate_password_hash("neuroscan_timing_mitigation_token_2026")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        if not email or not password:
            flash('Both email and password are required to sign in.', 'error')
            return render_template('login.html'), 400

        user = User.query.filter_by(email=email).first()
        # Timing-attack safe password comparison
        candidate_hash = user.password if user else DUMMY_PASSWORD_HASH
        is_valid_password = check_password_hash(candidate_hash, password)

        if user and is_valid_password:
            session.clear()
            session.permanent    = True
            session['user_id']   = user.id
            session['user_name'] = user.name
            return redirect(url_for('dashboard'))
        flash('Invalid email or password.', 'error')
    return render_template('login.html')

EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$')

@app.route('/register', methods=['POST'])
def register():
    name     = request.form.get('name', '').strip()
    email    = request.form.get('email', '').strip().lower()
    password = request.form.get('password', '')
    
    if not name:
        flash('Full name is required.', 'error')
        return redirect(url_for('login'))

    if len(password) > 128:
        flash('Password cannot exceed 128 characters.', 'error')
        return redirect(url_for('login'))
        return redirect(url_for('login'))
        
    if not email or not EMAIL_REGEX.match(email):
        flash('Please enter a valid clinical email address.', 'error')
        return redirect(url_for('login'))
        
    if User.query.filter_by(email=email).first():
        flash('Email already registered. Please sign in.', 'error')
        return redirect(url_for('login'))
        
    if len(password) < 8:
        flash('Password must be at least 8 characters for security compliance.', 'error')
        return redirect(url_for('login'))
        
    user = User(name=name, email=email, password=generate_password_hash(password))
    db.session.add(user)
    db.session.commit()
    flash('Account created! Please sign in.', 'success')
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ==============================
# Routes — Dashboard
# ==============================
@app.route('/dashboard')
@login_required
def dashboard():
    user          = current_user()
    total_scans    = Scan.query.filter_by(user_id=user.id).count()
    non_demented   = Scan.query.filter_by(user_id=user.id, prediction='NonDemented').count()
    mild_count     = Scan.query.filter(Scan.user_id == user.id, Scan.prediction.in_(['MildDemented', 'VeryMildDemented'])).count()
    moderate_count = Scan.query.filter_by(user_id=user.id, prediction='ModerateDemented').count()
    scans         = Scan.query.filter_by(user_id=user.id).order_by(Scan.created_at.desc()).limit(15).all()
    chat_sessions = ChatSession.query.filter_by(user_id=user.id)\
                                     .order_by(ChatSession.created_at.desc()).limit(10).all()
    return render_template('dashboard.html',
        current_user=user, scans=scans, chat_sessions=chat_sessions,
        total_scans=total_scans, non_demented=non_demented,
        mild_count=mild_count, moderate_count=moderate_count)

# ==============================
# Routes — Chat History
# ==============================
@app.route('/chat_history/<int:session_id>')
@login_required
def chat_history_view(session_id):
    user      = current_user()
    chat_sess = ChatSession.query.get_or_404(session_id)
    if chat_sess.user_id != user.id:
        return "Unauthorized", 403
    messages = ChatMessage.query.filter_by(chat_session_id=session_id)\
                                .order_by(ChatMessage.created_at).all()
    return render_template('chat_history.html',
        current_user=user, chat_session=chat_sess, messages=messages)

# ==============================
# Routes — Main App
# ==============================
@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    user = current_user()
    filename = prediction = confidence = all_probs = warning = description = risk = elapsed_ms = model_version = None

    if request.method == 'POST':
        if 'file' not in request.files or not request.files['file'].filename:
            flash("Please select an MRI image file to upload.", "error")
            return redirect(request.url)
        file = request.files['file']

        if not allowed_file(file.filename):
            flash(f"Unsupported file format. Permitted types: {', '.join(sorted(ALLOWED_EXTENSIONS)).upper()}", "error")
            return redirect(request.url)

        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        unique_prefix = uuid.uuid4().hex[:12]
        filename_secure = f"{unique_prefix}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename_secure)
        file.save(filepath)
        filename = filename_secure

        # Deep validation of image header and dimensions
        is_valid, validation_info = validate_mri_image(filepath)
        if not is_valid:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception:
                    pass
            flash(validation_info, "error")
            return redirect(request.url)

        try:
            in_channels = getattr(alz_model, 'in_channels', 3)
            try:
                image_tensor = transform_image(filepath, in_channels=in_channels)
            except TypeError:
                image_tensor = transform_image(filepath)

            pred_out = predict(alz_model, image_tensor, classes)
            if len(pred_out) == 5:
                prediction, confidence, all_probs, warning, elapsed_ms = pred_out
            else:
                prediction, confidence, all_probs, warning = pred_out[:4]
                elapsed_ms = 0.0

            description = stage_info.get(prediction, '')
            risk        = stage_risk.get(prediction, 'unknown')
            model_version = f"ResNet18-{in_channels}ch"

            scan = Scan(
                user_id       = user.id,
                prediction    = prediction,
                confidence    = confidence,
                all_probs     = json.dumps(all_probs),
                warning       = warning,
                processing_ms = elapsed_ms,
                model_version = model_version
            )
            db.session.add(scan)
            db.session.commit()

            session['stage']      = prediction
            session['confidence'] = confidence
            session['all_probs']  = all_probs
            session['scan_db_id'] = scan.id
            session.pop('chat_session_id', None)

        except Exception as e:
            app.logger.error(f"Inference processing failed: {e}", exc_info=True)
            if request.path.startswith('/api/') or request.is_json:
                return api_error(f"Diagnostic inference failed: {str(e)}", 500)
            flash(f"Diagnostic inference could not be completed: {str(e)}", "error")
            return redirect(request.url)
        finally:
            safe_remove(filepath)

    chart_dataset = prepare_chart_dataset(all_probs, predicted_stage=prediction)
    display_prediction = format_stage_label(prediction) if prediction else None

    return render_template('index.html',
        current_user=user,
        filename=filename, prediction=prediction, display_prediction=display_prediction,
        confidence=confidence, all_probs=all_probs, chart_dataset=chart_dataset,
        warning=warning, description=description, risk=risk,
        elapsed_ms=elapsed_ms, model_version=model_version)


def api_error(message, status_code=400, details=None):
    """Standardized JSON error responder for RESTful API endpoints."""
    payload = {
        "error": message,
        "status": status_code,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    if details:
        payload["details"] = details
    return jsonify(payload), status_code


@app.route('/chat', methods=['POST'])
@app.route('/api/chat', methods=['POST'])
@login_required
def chat():
    if not request.is_json:
        return api_error('Request payload must be JSON', 400)
    user_message = request.json.get('message', '').strip()
    if not user_message:
        return api_error('Message content cannot be empty', 400)
    response = generate_chatbot_response(user_message)
    return jsonify({'response': response, 'status': 'success', 'timestamp': datetime.utcnow().isoformat() + 'Z'}), 200


@app.route('/api/scans', methods=['GET'])
@login_required
def api_get_scans():
    """Retrieve the authenticated clinician's scan records as JSON."""
    user = current_user()
    scans = Scan.query.filter_by(user_id=user.id).order_by(Scan.created_at.desc()).limit(50).all()
    results = []
    for s in scans:
        try:
            probs = json.loads(s.all_probs) if s.all_probs else {}
        except Exception:
            probs = {}
        results.append({
            'id': s.id,
            'prediction': s.prediction,
            'confidence': s.confidence,
            'all_probs': probs,
            'risk_tier': get_risk_tier(s.prediction),
            'processing_ms': getattr(s, 'processing_ms', 0.0),
            'model_version': getattr(s, 'model_version', 'ResNet18'),
            'created_at': s.created_at.isoformat() if s.created_at else None
        })
    return jsonify({'scans': results, 'count': len(results)}), 200


@app.route('/api/scans/<int:scan_id>', methods=['GET'])
@login_required
def api_get_scan(scan_id):
    """Retrieve details for a specific diagnostic scan by ID."""
    user = current_user()
    scan = Scan.query.get(scan_id)
    if not scan:
        return api_error('Scan record not found', 404)
    if scan.user_id != user.id:
        return api_error('Unauthorized to access this scan record', 403)
    try:
        probs = json.loads(scan.all_probs) if scan.all_probs else {}
    except Exception:
        probs = {}
    return jsonify({
        'id': scan.id,
        'prediction': scan.prediction,
        'confidence': scan.confidence,
        'all_probs': probs,
        'risk_tier': get_risk_tier(scan.prediction),
        'warning': scan.warning,
        'processing_ms': getattr(scan, 'processing_ms', 0.0),
        'model_version': getattr(scan, 'model_version', 'ResNet18'),
        'created_at': scan.created_at.isoformat() if scan.created_at else None
    }), 200

@app.route('/clear_chat', methods=['POST'])
@login_required
def clear_chat():
    session.pop('chat_session_id', None)
    return jsonify({'status': 'cleared'})

@app.route('/download_report')
@login_required
def download_report():
    user       = current_user()
    stage      = session.get('stage')
    confidence = session.get('confidence')
    all_probs  = session.get('all_probs')
    if not stage or not confidence or not all_probs:
        return 'No prediction found. Please upload a scan first.', 400

    unique_suffix = f"{user.id}_{uuid.uuid4().hex[:8]}"
    report_path = os.path.join(app.config['REPORTS_FOLDER'], f"report_{unique_suffix}.pdf")
    generate_report(
        prediction=stage,
        confidence=confidence,
        all_probs=all_probs,
        output_path=report_path,
        model_version=session.get('model_version', 'ResNet18-Adaptive'),
        reference_id=f"NS-{unique_suffix.upper()}"
    )
    return send_file(report_path, as_attachment=True,
                     download_name=f'neuroscan_report_{stage}.pdf',
                     mimetype='application/pdf')

@app.route('/download_report/<int:scan_id>')
@login_required
def download_report_by_id(scan_id):
    user = current_user()
    scan = Scan.query.get_or_404(scan_id)
    if scan.user_id != user.id:
        return 'Unauthorized', 403
    all_probs   = json.loads(scan.all_probs)
    report_path = os.path.join(app.config['REPORTS_FOLDER'], f'report_{scan.id}_{scan.user_id}.pdf')
    generate_report(
        prediction=scan.prediction,
        confidence=scan.confidence,
        all_probs=all_probs,
        output_path=report_path,
        model_version=scan.model_version or "ResNet18-Adaptive",
        reference_id=f"NS-SCAN-{scan.id:06d}"
    )
    return send_file(report_path, as_attachment=True,
                     download_name=f'neuroscan_report_scan_{scan_id}.pdf',
                     mimetype='application/pdf')

# ==============================
# Routes — Reports
# ==============================
@app.route('/reports')
@login_required
def reports():
    user  = current_user()
    scans = Scan.query.filter_by(user_id=user.id).order_by(Scan.created_at.desc()).all()
    return render_template('reports.html', current_user=user, scans=scans)

# ==============================
# Routes — Scan History
# ==============================
@app.route('/history')
@login_required
def history():
    user  = current_user()
    page  = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 12, type=int)
    pagination = Scan.query.filter_by(user_id=user.id).order_by(Scan.created_at.desc()).paginate(page=page, per_page=per_page, error_out=False)
    return render_template('history.html', current_user=user, scans=pagination.items, pagination=pagination)

@app.route('/delete_scan/<int:scan_id>', methods=['POST'])
@login_required
def delete_scan(scan_id):
    user = current_user()
    scan = Scan.query.get_or_404(scan_id)
    if scan.user_id != user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    for cs in ChatSession.query.filter_by(scan_id=scan_id).all():
        ChatMessage.query.filter_by(chat_session_id=cs.id).delete()
        db.session.delete(cs)
    db.session.delete(scan)
    db.session.commit()
    return jsonify({'status': 'deleted'})

# ==============================
# Routes — Chat History List
# ==============================
@app.route('/chat_history')
@login_required
def chat_history_list():
    user          = current_user()
    chat_sessions = ChatSession.query.filter_by(user_id=user.id)\
                                     .order_by(ChatSession.created_at.desc()).all()
    return render_template('chat_history_list.html', current_user=user, chat_sessions=chat_sessions)

@app.route('/delete_chat/<int:session_id>', methods=['POST'])
@login_required
def delete_chat(session_id):
    user      = current_user()
    chat_sess = ChatSession.query.get_or_404(session_id)
    if chat_sess.user_id != user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    ChatMessage.query.filter_by(chat_session_id=session_id).delete()
    db.session.delete(chat_sess)
    db.session.commit()
    return jsonify({'status': 'deleted'})

@app.route('/uploads/<path:filename>')
@login_required
def serve_upload(filename):
    safe_name = secure_filename(os.path.basename(filename))
    if not safe_name:
        return "Invalid filename", 400
    safe_dir = os.path.abspath(app.config['UPLOAD_FOLDER'])
    target_path = os.path.abspath(os.path.join(safe_dir, safe_name))
    if not target_path.startswith(safe_dir) or not os.path.exists(target_path):
        return "File not found or access denied", 404
    return send_file(target_path)

# ==============================
# Security Middleware (OWASP)
# ==============================
@app.after_request
def set_security_headers(response):
    """Enforces essential OWASP security headers on all HTTP responses."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
    csp = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdnjs.cloudflare.com; "
        "font-src 'self' https://fonts.gstatic.com https://cdnjs.cloudflare.com; "
        "img-src 'self' data: blob:; "
        "connect-src 'self';"
    )
    response.headers['Content-Security-Policy'] = csp
    # Performance cache headers: static assets cached aggressively, dynamic endpoints revalidated
    if request.path.startswith('/static/'):
        response.headers['Cache-Control'] = 'public, max-age=31536000, immutable'
    elif 'Cache-Control' not in response.headers:
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response


# ==============================
# Clinical Error Handlers
# ==============================
@app.errorhandler(400)
def handle_bad_request(e):
    app.logger.warning(f"400 Bad Request: {request.path} - {e}")
    if request.path.startswith('/api/') or request.is_json:
        return jsonify({"error": "Bad Request", "message": str(getattr(e, 'description', e)), "status": 400}), 400
    return render_template('errors/400.html', error_message=getattr(e, 'description', None)), 400


@app.errorhandler(403)
def handle_forbidden(e):
    app.logger.warning(f"403 Forbidden: {request.path} - {e}")
    if request.path.startswith('/api/') or request.is_json:
        return jsonify({"error": "Forbidden", "message": "Access denied.", "status": 403}), 403
    return render_template('errors/403.html', error_message=getattr(e, 'description', None)), 403


@app.errorhandler(404)
def handle_not_found(e):
    app.logger.info(f"404 Not Found: {request.path}")
    if request.path.startswith('/api/') or request.is_json:
        return jsonify({"error": "Not Found", "message": "Resource not found.", "status": 404}), 404
    return render_template('errors/404.html', error_message=getattr(e, 'description', None)), 404


@app.errorhandler(413)
def handle_large_payload(e):
    app.logger.warning(f"413 Payload Too Large: {request.path}")
    if request.path.startswith('/api/') or request.is_json:
        return jsonify({"error": "Payload Too Large", "message": "Uploaded scan exceeds the 16 MB limit.", "status": 413}), 413
    flash("The uploaded file exceeds the maximum allowed size of 16 MB.", "error")
    return redirect(url_for('index'))


@app.errorhandler(500)
def handle_internal_error(e):
    app.logger.error(f"500 Internal Error: {request.path} - {e}", exc_info=True)
    if request.path.startswith('/api/') or request.is_json:
        return jsonify({"error": "Internal Server Error", "message": "An unexpected diagnostic error occurred.", "status": 500}), 500
    return render_template('errors/500.html'), 500


# ==============================
# Run
# ==============================
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)

