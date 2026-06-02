import os
from dotenv import load_dotenv

load_dotenv()
from groq import Groq

def main():
    key = os.getenv('GROQ_API_KEY')
    print('GROQ_API_KEY present:', bool(key))
    try:
        client = Groq(api_key=key)
        # Minimal completion request
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello."}
        ]
        resp = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages, max_tokens=20)
        print('Response keys:', resp.keys() if hasattr(resp, 'keys') else type(resp))
        print('Done')
    except Exception as e:
        print('Groq exception:', repr(e))

if __name__ == '__main__':
    main()
