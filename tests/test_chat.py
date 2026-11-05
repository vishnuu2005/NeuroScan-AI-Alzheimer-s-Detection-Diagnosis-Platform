import uuid
import pytest
from chatbot.app import app, db, User, ChatSession, ChatMessage
from chatbot.clinical_ai import generate_local_clinical_response, GDS_STAGES

@pytest.fixture
def auth_client():
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    with app.test_client() as client:
        with app.app_context():
            unique_email = f"chat_user_{uuid.uuid4().hex[:8]}@clinic.org"
            user = User(name="Chat Tester", email=unique_email, password="SecurePassword123!")
            db.session.add(user)
            db.session.commit()
            user_id = user.id

        with client.session_transaction() as sess:
            sess['user_id'] = user_id
            sess['stage'] = "VeryMildDemented"
            sess['confidence'] = 84.5

        yield client, user_id

        with app.app_context():
            u = User.query.get(user_id)
            if u:
                sessions = ChatSession.query.filter_by(user_id=user_id).all()
                for s in sessions:
                    ChatMessage.query.filter_by(chat_session_id=s.id).delete()
                    db.session.delete(s)
                db.session.delete(u)
                db.session.commit()

def test_local_clinical_response_symptoms():
    resp = generate_local_clinical_response(
        user_message="What symptoms are typical for this stage?",
        stage="VeryMildDemented",
        confidence=85.0
    )
    assert "## Clinical Symptom Profile" in resp
    assert "Very Mild" in resp or "Mild Cognitive Impairment" in resp

def test_local_clinical_response_treatment():
    resp = generate_local_clinical_response(
        user_message="Are there treatments or medications available?",
        stage="MildDemented",
        confidence=90.0
    )
    assert "Therapeutic & Management Overview" in resp
    assert "Cholinesterase Inhibitors" in resp

def test_local_clinical_response_lifestyle():
    resp = generate_local_clinical_response(
        user_message="What diet and exercise do you recommend for brain health?",
        stage="NonDemented",
        confidence=95.0
    )
    assert "Brain Health & Neuroprotective Guidance" in resp
    assert "MIND" in resp or "Mediterranean" in resp

def test_local_clinical_response_caregiver():
    resp = generate_local_clinical_response(
        user_message="I need help with caregiver safety and family support.",
        stage="ModerateDemented",
        confidence=88.0
    )
    assert "Caregiver Support & Safety Framework" in resp
    assert "Home Safety" in resp

def test_chat_unauthenticated_blocked():
    with app.test_client() as client:
        resp = client.post('/chat', json={'message': 'Hello assistant'})
        assert resp.status_code in [302, 401]

def test_chat_empty_payload_rejected(auth_client):
    client, _ = auth_client
    resp = client.post('/chat', json={'message': '   '})
    assert resp.status_code == 400

def test_chat_authenticated_conversation_and_db_persistence(auth_client):
    client, user_id = auth_client
    payload = {'message': 'What are the main symptoms of this MRI classification?'}
    resp = client.post('/chat', json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'response' in data
    assert data['status'] == 'success'
    assert len(data['response']) > 20

    with app.app_context():
        chat_sess = ChatSession.query.filter_by(user_id=user_id).first()
        assert chat_sess is not None
        msgs = ChatMessage.query.filter_by(chat_session_id=chat_sess.id).all()
        assert len(msgs) == 2
        assert msgs[0].role == 'user'
        assert msgs[1].role == 'assistant'

def test_clear_chat_endpoint(auth_client):
    client, _ = auth_client
    resp = client.post('/clear_chat')
    assert resp.status_code == 200
    assert resp.get_json() == {'status': 'cleared'}
