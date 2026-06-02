import os
import json
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, send_file, flash
from PIL import Image
import torch
import torchvision.transforms as transforms
from .inference import load_model, predict, transform_image   # import transform_image from inference
from dotenv import load_dotenv
from groq import Groq
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from .generate_reports import generate_report

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
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "neuroscan_secret_2024")
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///neuroscan.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ==============================
# Database Models
# ==============================
class User(db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    name          = db.Column(db.String(100), nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    password      = db.Column(db.String(200), nullable=False)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)
    scans         = db.relationship('Scan', backref='user', lazy=True)
    chat_sessions = db.relationship('ChatSession', backref='user', lazy=True)

class Scan(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    prediction  = db.Column(db.String(50), nullable=False)
    confidence  = db.Column(db.Float, nullable=False)
    all_probs   = db.Column(db.Text, nullable=False)
    warning     = db.Column(db.String(200))
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)

class ChatSession(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    scan_id    = db.Column(db.Integer, db.ForeignKey('scan.id'), nullable=True)
    scan_stage = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    messages   = db.relationship('ChatMessage', backref='chat_session', lazy=True)

    @property
    def message_count(self):
        return len(self.messages)

class ChatMessage(db.Model):
    id              = db.Column(db.Integer, primary_key=True)
    chat_session_id = db.Column(db.Integer, db.ForeignKey('chat_session.id'), nullable=False)
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
# Resolve model path robustly: prefer configured absolute path, else fall back to project models folder
possible_model_paths = [
    "D:/AlzeigmersChatBot/AlzeigmersChatBot/models/resnet2d_alzheimers.pth",
    os.path.join(os.path.dirname(__file__), "..", "models", "resnet2d_alzheimers.pth"),
    os.path.join(os.getcwd(), "models", "resnet2d_alzheimers.pth"),
    os.path.join(os.getcwd(), "AlzeigmersChatBot", "models", "resnet2d_alzheimers.pth"),
    "models/resnet2d_alzheimers.pth"
]
model_path = None
for p in possible_model_paths:
    p_norm = os.path.normpath(p)
    if os.path.exists(p_norm):
        model_path = p_norm
        break
if model_path is None:
    raise FileNotFoundError("resnet2d_alzheimers.pth not found in expected locations.\n"
                            "Checked: " + ", ".join(possible_model_paths))
alz_model = load_model(model_path, num_classes=len(classes))

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
    if session_id:
        chat_sess = ChatSession.query.get(session_id)
        if chat_sess and chat_sess.user_id == user.id:
            return chat_sess
    scan_id    = session.get('scan_db_id')
    scan_stage = session.get('stage')
    chat_sess  = ChatSession(user_id=user.id, scan_id=scan_id, scan_stage=scan_stage)
    db.session.add(chat_sess)
    db.session.commit()
    session['chat_session_id'] = chat_sess.id
    return chat_sess

def generate_stage_context():
    stage      = session.get('stage')
    confidence = session.get('confidence')
    if stage and confidence is not None:
        advice = {
            "NonDemented":      "The MRI suggests no signs of dementia. Provide reassurance and preventive lifestyle guidance.",
            "VeryMildDemented": "Early signs detected. Discuss early symptoms, monitoring, and lifestyle changes.",
            "MildDemented":     "Mild cognitive decline. Provide structured guidance and caregiver planning advice.",
            "ModerateDemented": "Moderate progression. Focus on caregiving strategies and medical supervision.",
            "Uncertain":        "The model could not confidently classify the scan. Advise professional evaluation."
        }.get(stage, "")
        return f"The MRI prediction stage is {stage} with {confidence:.2f}% confidence. {advice}"
    return "No MRI scan has been uploaded yet."

def generate_chatbot_response(user_message):
    system_prompt = """You are a professional, compassionate medical AI assistant specialising in Alzheimer's disease.
Rules:
- NEVER provide a final medical diagnosis.
- Always recommend consulting a qualified neurologist.
- Be structured, clear, and calm.
- Do not fabricate statistics.
Respond using markdown headings (##), bullet points, and bold text where appropriate."""

    stage_context = generate_stage_context()
    chat_sess     = get_or_create_chat_session()

    history = []
    if chat_sess:
        msgs = ChatMessage.query.filter_by(chat_session_id=chat_sess.id)\
                                .order_by(ChatMessage.created_at.desc())\
                                .limit(10).all()
        for m in reversed(msgs):
            history.append({"role": m.role, "content": m.content})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": stage_context},
        *history,
        {"role": "user", "content": user_message}
    ]

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.6,
            max_tokens=1024
        )
        reply = completion.choices[0].message.content.strip()
        if chat_sess:
            db.session.add(ChatMessage(chat_session_id=chat_sess.id, role='user',      content=user_message))
            db.session.add(ChatMessage(chat_session_id=chat_sess.id, role='assistant', content=reply))
            db.session.commit()
        return reply
    except Exception as e:
        print("Groq error:", e)
        # Local fallback response when external chat service fails
        advice = stage_info.get(session.get('stage'), '')
        fallback = (
            "## Assistant\n\n"
            "**Note:** The external chatbot service is currently unavailable.\n\n"
            f"**Context:** {stage_context}\n\n"
            "**Summary:** I can provide general, non-diagnostic guidance based on the MRI stage above.\n\n"
            f"{advice}\n\n"
            "**Recommendations:**\n"
            "- Do not treat this as a medical diagnosis.\n"
            "- Consult a qualified neurologist for detailed assessment and next steps.\n"
            "- If symptoms are worsening, seek immediate medical attention.\n\n"
            "If you'd like, try re-sending your message later or upload a clearer scan."
        )
        if chat_sess:
            db.session.add(ChatMessage(chat_session_id=chat_sess.id, role='user',      content=user_message))
            db.session.add(ChatMessage(chat_session_id=chat_sess.id, role='assistant', content=fallback))
            db.session.commit()
        return fallback

# ==============================
# Routes — Auth
# ==============================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        user     = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session.clear()
            session['user_id']   = user.id
            session['user_name'] = user.name
            return redirect(url_for('dashboard'))
        flash('Invalid email or password.', 'error')
    return render_template('login.html')

@app.route('/register', methods=['POST'])
def register():
    name     = request.form.get('name', '').strip()
    email    = request.form.get('email', '').strip().lower()
    password = request.form.get('password', '')
    if User.query.filter_by(email=email).first():
        flash('Email already registered.', 'error')
        return redirect(url_for('login'))
    if len(password) < 6:
        flash('Password must be at least 6 characters.', 'error')
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
    scans         = Scan.query.filter_by(user_id=user.id).order_by(Scan.created_at.desc()).all()
    chat_sessions = ChatSession.query.filter_by(user_id=user.id)\
                                     .order_by(ChatSession.created_at.desc()).all()
    total_scans    = len(scans)
    non_demented   = sum(1 for s in scans if s.prediction == 'NonDemented')
    mild_count     = sum(1 for s in scans if s.prediction in ['MildDemented', 'VeryMildDemented'])
    moderate_count = sum(1 for s in scans if s.prediction == 'ModerateDemented')
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
    filename = prediction = confidence = all_probs = warning = description = risk = None

    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            return redirect(request.url)
        file = request.files['file']
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filename_secure = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename_secure)
        file.save(filepath)
        filename = filename_secure
        try:
            image_tensor = transform_image(filepath)   # ✅ uses inference.py's version
            prediction, confidence, all_probs, warning = predict(alz_model, image_tensor, classes)
            description = stage_info.get(prediction, '')
            risk        = stage_risk.get(prediction, 'unknown')

            scan = Scan(
                user_id    = user.id,
                prediction = prediction,
                confidence = confidence,
                all_probs  = json.dumps(all_probs),
                warning    = warning
            )
            db.session.add(scan)
            db.session.commit()

            session['stage']      = prediction
            session['confidence'] = confidence
            session['all_probs']  = all_probs
            session['scan_db_id'] = scan.id
            session.pop('chat_session_id', None)

        except Exception as e:
            return f"Error processing image: {str(e)}", 500
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    return render_template('index.html',
        current_user=user,
        filename=filename, prediction=prediction, confidence=confidence,
        all_probs=all_probs, warning=warning, description=description, risk=risk)

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    user_message = request.json.get('message')
    response     = generate_chatbot_response(user_message)
    return jsonify({'response': response})

@app.route('/clear_chat', methods=['POST'])
@login_required
def clear_chat():
    session.pop('chat_session_id', None)
    return jsonify({'status': 'cleared'})

@app.route('/download_report')
@login_required
def download_report():
    stage      = session.get('stage')
    confidence = session.get('confidence')
    all_probs  = session.get('all_probs')
    if not stage or not confidence or not all_probs:
        return 'No prediction found. Please upload a scan first.', 400
    report_path = 'static/reports/alzheimers_report.pdf'
    os.makedirs('static/reports', exist_ok=True)
    generate_report(stage, confidence, all_probs, report_path)
    return send_file(report_path, as_attachment=True,
                     download_name='alzheimers_mri_report.pdf',
                     mimetype='application/pdf')

@app.route('/download_report/<int:scan_id>')
@login_required
def download_report_by_id(scan_id):
    user = current_user()
    scan = Scan.query.get_or_404(scan_id)
    if scan.user_id != user.id:
        return 'Unauthorized', 403
    all_probs   = json.loads(scan.all_probs)
    report_path = f'static/reports/report_{scan_id}.pdf'
    os.makedirs('static/reports', exist_ok=True)
    generate_report(scan.prediction, scan.confidence, all_probs, report_path)
    return send_file(report_path, as_attachment=True,
                     download_name=f'alzheimers_report_{scan_id}.pdf',
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
    scans = Scan.query.filter_by(user_id=user.id).order_by(Scan.created_at.desc()).all()
    return render_template('history.html', current_user=user, scans=scans)

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

@app.route('/uploads/<filename>')
def serve_upload(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

# ==============================
# Run
# ==============================
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)

