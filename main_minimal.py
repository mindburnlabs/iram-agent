#!/usr/bin/env python3
"""
Minimal test version of IRAM to verify deployment works
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime

# Create minimal FastAPI app
app = FastAPI(
    title="IRAM - Minimal Test Server",
    description="Minimal version for testing deployment",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "IRAM Minimal Test Server",
        "version": "0.1.0"
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "IRAM Minimal Test Server is running",
        "endpoints": {
            "health": "/health",
            "docs": "/docs"
        }
    }

def main():
    """Main function to run the server."""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    print(f"Starting IRAM Minimal Server on {host}:{port}")
    
    uvicorn.run(
        "main_minimal:app",
        host=host,
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    main()