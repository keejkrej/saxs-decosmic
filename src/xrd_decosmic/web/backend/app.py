"""
FastAPI backend for XRD Decosmic web application.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .routers import api

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="XRD Decosmic API")

    # Configure CORS for development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],  # Vite's default port
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(api.router, prefix="/api")

    return app

app = create_app()

def start_server(host: str = "localhost", port: int = 8000):
    """Start the FastAPI server."""
    uvicorn.run(app, host=host, port=port) 