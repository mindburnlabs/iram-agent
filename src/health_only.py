#!/usr/bin/env python3
"""
Ultra minimal health check app for debugging Railway deployment issues
"""

from fastapi import FastAPI
import uvicorn
import os

# Create minimal FastAPI app
app = FastAPI(title="IRAM Health Check")

@app.get("/health")
def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "service": "iram-health-check",
        "port": os.getenv("PORT", "8000")
    }

@app.get("/")
def root():
    """Root endpoint."""
    return {"message": "IRAM Health Check Service"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)