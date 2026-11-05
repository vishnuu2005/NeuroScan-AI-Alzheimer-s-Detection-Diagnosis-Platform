"""
Production WSGI Entrypoint for NeuroScan AI.
Supports Waitress (recommended for Windows 11 / multi-threaded CPU inference),
Gunicorn (for Linux production containers), and standard WSGI servers.
"""
import os
import sys
import argparse
from chatbot.app import app

# Standard WSGI callable alias
application = app


def parse_args():
    parser = argparse.ArgumentParser(description="NeuroScan AI Production WSGI Server")
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"), help="Host IP address to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "5000")), help="Port to bind (default: 5000)")
    parser.add_argument("--threads", type=int, default=int(os.getenv("WSGI_THREADS", "4")), help="Worker threads count (default: 4)")
    return parser.parse_args()


def run_production_server():
    args = parse_args()
    print(f"============================================================")
    print(f" NeuroScan AI — Clinical Production Server Starting")
    print(f" Listening on: http://{args.host}:{args.port}")
    print(f" Worker Threads: {args.threads}")
    print(f" Platform: {sys.platform}")
    print(f"============================================================")

    # Attempt Waitress serving first (cross-platform, native on Windows)
    try:
        from waitress import serve
        serve(app, host=args.host, port=args.port, threads=args.threads)
        return
    except ImportError:
        pass

    # Fallback to standard Flask runner if waitress is not installed
    print("[WARNING] Waitress is not installed. Falling back to built-in WSGI server.")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    run_production_server()
