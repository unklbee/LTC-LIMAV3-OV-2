# src/server/main.py

"""
LIMA Traffic Counter - Web Server Module

This module provides a web API server for receiving and managing
traffic count data from multiple camera installations.
"""

import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from ..services.database_service import create_database_service, CountRecord
from ..object_detector.config import Config

logger = logging.getLogger(__name__)

# Pydantic models for API
class CountDataModel(BaseModel):
    """Model for incoming count data from cameras."""
    host_id: str
    interval_start: str  # ISO format datetime
    interval_end: str    # ISO format datetime
    counts: Dict[str, int]  # vehicle type -> count

class CountResponseModel(BaseModel):
    """Response model for count operations."""
    success: bool
    message: str
    record_id: Optional[int] = None

class HealthResponseModel(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    database_connected: bool
    active_hosts: List[str]

# FastAPI app
app = FastAPI(
    title="LIMA Traffic Counter API",
    description="API server for traffic counting data collection",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory tracking of active hosts
active_hosts = set()

def get_database_service():
    """Dependency to get database service."""
    try:
        return create_database_service()
    except Exception as e:
        logger.error(f"Failed to create database service: {e}")
        raise HTTPException(status_code=500, detail="Database service unavailable")

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with basic info."""
    return {
        "service": "LIMA Traffic Counter API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "submit_counts": "/api/v1/counts",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponseModel)
async def health_check(db_service = Depends(get_database_service)):
    """Health check endpoint."""
    try:
        db_available = db_service.is_available()
        return HealthResponseModel(
            status="healthy" if db_available else "degraded",
            timestamp=datetime.now().isoformat(),
            database_connected=db_available,
            active_hosts=list(active_hosts)
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponseModel(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            database_connected=False,
            active_hosts=[]
        )

@app.post("/api/v1/counts", response_model=CountResponseModel)
async def submit_counts(
        count_data: CountDataModel,
        db_service = Depends(get_database_service)
):
    """Submit traffic count data from a camera installation."""
    try:
        # Parse datetime strings
        interval_start = datetime.fromisoformat(count_data.interval_start.replace('Z', '+00:00'))
        interval_end = datetime.fromisoformat(count_data.interval_end.replace('Z', '+00:00'))

        # Convert vehicle names to class IDs
        class_name_to_id = {
            "bicycle": 1,
            "car": 2,
            "motorbike": 3,
            "bus": 5,
            "truck": 7
        }

        # Convert counts from names to class IDs
        counts_by_id = {}
        for vehicle_name, count in count_data.counts.items():
            class_id = class_name_to_id.get(vehicle_name.lower())
            if class_id is not None:
                counts_by_id[class_id] = count

        # Create count record
        record = CountRecord.from_counts_dict(
            host_id=count_data.host_id,
            interval_start=interval_start,
            interval_end=interval_end,
            counts=counts_by_id
        )

        # Save to database
        success = db_service.save_count_record(record)

        if success:
            # Track active host
            active_hosts.add(count_data.host_id)

            logger.info(f"Received counts from {count_data.host_id}: {count_data.counts}")
            return CountResponseModel(
                success=True,
                message="Count data saved successfully",
                record_id=record.id
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to save count data")

    except ValueError as e:
        logger.error(f"Invalid data format: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid data format: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing count data: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/hosts", response_model=List[str])
async def get_active_hosts():
    """Get list of active camera hosts."""
    return list(active_hosts)

@app.delete("/api/v1/hosts/{host_id}")
async def remove_host(host_id: str):
    """Remove a host from active tracking."""
    if host_id in active_hosts:
        active_hosts.remove(host_id)
        return {"message": f"Host {host_id} removed from active tracking"}
    else:
        raise HTTPException(status_code=404, detail="Host not found in active tracking")

def setup_logging():
    """Setup logging configuration for the server."""
    logging.basicConfig(
        level=Config.get_log_level(),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('lima_server.log')
        ]
    )

def main():
    """Main entry point for the server."""
    setup_logging()

    # Ensure database directory exists
    Config.ensure_dirs()

    logger.info("Starting LIMA Traffic Counter API Server")

    # Configuration
    host = "0.0.0.0"
    port = 8000

    # Check if running in development mode
    import sys
    debug = "--debug" in sys.argv or "--reload" in sys.argv

    if debug:
        logger.info("Running in debug mode with auto-reload")
        uvicorn.run(
            "src.server.main:app",
            host=host,
            port=port,
            reload=True,
            log_level="info"
        )
    else:
        logger.info(f"Starting production server on {host}:{port}")
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="warning"
        )

if __name__ == "__main__":
    main()