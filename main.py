"""Main entry point for GramBrain API server."""

import uvicorn
import sys


def main():
    """Run the API server."""
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
