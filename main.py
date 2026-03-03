"""Main entry point for GramBrain API server."""

import uvicorn
from fastapi import FastAPI
from backend.src.api.routes import app as api_app


# Create main app
app = FastAPI(
    title="GramBrain AI",
    description="Multi-Agent Agricultural Intelligence Platform",
    version="0.1.0",
)

# Mount API routes under /api prefix
app.mount("/api", api_app)


def main():
    """Run the API server."""
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
