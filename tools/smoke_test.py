import requests
from pathlib import Path

BASE = "http://127.0.0.1:5000"

def main():
    s = requests.Session()
    email = "tester@example.com"
    password = "password123"

    # 1) Register (ignore if already exists)
    try:
        r = s.post(f"{BASE}/register", data={
            'name': 'Smoke Tester', 'email': email, 'password': password
        }, allow_redirects=True, timeout=15)
        print('Register status:', r.status_code)
    except Exception as e:
        print('Register error:', e)

    # 2) Login
    r = s.post(f"{BASE}/login", data={'email': email, 'password': password}, allow_redirects=True, timeout=15)
    print('Login status:', r.status_code)
    if r.status_code != 200:
        print('Login failed, response length:', len(r.text))

    # 3) Upload sample image
    sample = Path(__file__).parents[1] / 'chatbot' / 'static' / 'uploads' / 'd2.avif'
    if not sample.exists():
        print('Sample image not found:', sample)
        return

    with open(sample, 'rb') as f:
        files = {'file': (sample.name, f, 'image/avif')}
        r = s.post(f"{BASE}/", files=files, allow_redirects=True, timeout=60)
        print('Upload response status:', r.status_code)
        print('Upload response length:', len(r.text))

    # 4) Call chat
    try:
        r = s.post(f"{BASE}/chat", json={'message': 'Hello, I have questions about the scan.'}, timeout=20)
        print('/chat status:', r.status_code, 'json:', r.json())
    except Exception as e:
        print('Chat error:', e)

if __name__ == '__main__':
    main()
