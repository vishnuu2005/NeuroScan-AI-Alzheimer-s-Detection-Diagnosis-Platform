import json
import uuid
import pytest
from chatbot.app import app, db, User, Scan, ChatSession, ChatMessage

@pytest.fixture
def test_user():
    app.config['TESTING'] = True
    with app.app_context():
        unique_email = f"user_{uuid.uuid4().hex[:8]}@hospital.org"
        user = User(name="Database Test User", email=unique_email, password="HashedPassword123!")
        db.session.add(user)
        db.session.commit()
        yield user
        # Safe cleanup
        try:
            db.session.delete(user)
            db.session.commit()
        except Exception:
            db.session.rollback()

def test_user_creation_and_query(test_user):
    user = User.query.get(test_user.id)
    assert user is not None
    assert user.name == "Database Test User"
    assert "@hospital.org" in user.email

def test_scan_relationship_and_cascade(test_user):
    user = test_user
    scan = Scan(
        user_id=user.id,
        prediction="MildDemented",
        confidence=87.5,
        all_probs=json.dumps({"MildDemented": 87.5, "NonDemented": 12.5}),
        warning=None,
        processing_ms=124.5,
        model_version="ResNet18-Adaptive"
    )
    db.session.add(scan)
    db.session.commit()
    
    assert len(user.scans) >= 1
    assert scan.user.email == user.email
    assert scan.prediction == "MildDemented"
    assert scan.processing_ms == 124.5
    assert scan.model_version == "ResNet18-Adaptive"
    
    # Verify scan deletion
    scan_id = scan.id
    db.session.delete(scan)
    db.session.commit()
    assert Scan.query.get(scan_id) is None

def test_scan_pagination_and_aggregation(test_user):
    user = test_user
    # Insert multiple scans
    dummy_probs = json.dumps({"NonDemented": 90.0, "MildDemented": 10.0})
    scans_to_add = [
        Scan(user_id=user.id, prediction="NonDemented", confidence=95.0, all_probs=dummy_probs),
        Scan(user_id=user.id, prediction="VeryMildDemented", confidence=78.2, all_probs=dummy_probs),
        Scan(user_id=user.id, prediction="ModerateDemented", confidence=89.1, all_probs=dummy_probs),
    ]
    db.session.add_all(scans_to_add)
    db.session.commit()

    # Test count aggregation
    total = Scan.query.filter_by(user_id=user.id).count()
    assert total >= 3
    non_demented = Scan.query.filter_by(user_id=user.id, prediction="NonDemented").count()
    assert non_demented >= 1

    # Test pagination
    paginated = Scan.query.filter_by(user_id=user.id).paginate(page=1, per_page=2, error_out=False)
    assert len(paginated.items) == 2
    assert paginated.total >= 3
    assert paginated.has_next is True

    # Cleanup added scans
    for s in scans_to_add:
        db.session.delete(s)
    db.session.commit()

def test_chat_session_and_message_relationship(test_user):
    user = test_user
    chat_sess = ChatSession(user_id=user.id, scan_stage="VeryMildDemented")
    db.session.add(chat_sess)
    db.session.commit()
    
    msg1 = ChatMessage(chat_session_id=chat_sess.id, role="user", content="Hello assistant")
    msg2 = ChatMessage(chat_session_id=chat_sess.id, role="assistant", content="Hello clinician")
    db.session.add_all([msg1, msg2])
    db.session.commit()
    
    assert chat_sess.message_count == 2
    assert len(chat_sess.messages) == 2
    
    # Cleanup
    msg1_id = msg1.id
    db.session.delete(chat_sess)
    db.session.commit()
    assert ChatMessage.query.get(msg1_id) is None
