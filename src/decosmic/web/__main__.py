"""
Entry point for running the XRD Decosmic web application.
"""
import multiprocessing
import webbrowser
import time
import os
import signal
import sys
from pathlib import Path

def run_frontend():
    """Run the Vite development server."""
    frontend_dir = Path(__file__).parent / "frontend"
    os.chdir(frontend_dir)
    os.system("npm install")  # Install dependencies
    os.system("npm run dev")  # Start dev server

def run_backend():
    """Run the FastAPI backend server."""
    from . import start_server
    start_server()

def main():
    """Start both frontend and backend servers."""
    # Start backend server in a separate process
    backend = multiprocessing.Process(target=run_backend)
    backend.start()

    # Start frontend server in a separate process
    frontend = multiprocessing.Process(target=run_frontend)
    frontend.start()

    # Wait a bit for servers to start
    time.sleep(2)

    # Open browser
    webbrowser.open("http://localhost:5173")

    try:
        # Wait for processes to complete
        backend.join()
        frontend.join()
    except KeyboardInterrupt:
        # Handle graceful shutdown
        print("\nShutting down servers...")
        backend.terminate()
        frontend.terminate()
        backend.join()
        frontend.join()
        sys.exit(0)

if __name__ == "__main__":
    main() 