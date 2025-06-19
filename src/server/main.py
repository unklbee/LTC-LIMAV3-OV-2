# src/server/main.py
"""
REST API server for LIMA Traffic Counter.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
import uuid
import io
from pathlib import Path

from src.models.config import AppConfig
from src.services.database_service import create_database_service
from src.services.export_service import ExportService
from src.core.pipeline import create_pipeline

import structlog

logger = structlog.get_logger()

# API Models
class CountEvent(BaseModel):
    """Count event model"""
    timestamp: datetime
    line_id: str
    vehicle_type: str
    direction: str
    count: int = 1
    speed_kmh: Optional[float] = None
    confidence: Optional[float] = None


class CountingLine(BaseModel):
    """Counting line configuration"""
    id: str
    start: List[float]
    end: List[float]
    direction: Optional[List[float]] = None
    bidirectional: bool = True


class ProcessingRequest(BaseModel):
    """Video processing request"""
    video_url: Optional[str] = None
    counting_lines: List[CountingLine]
    roi_polygon: Optional[List[List[float]]] = None
    model: str = "yolov7-tiny"
    device: str = "AUTO"


class ExportRequest(BaseModel):
    """Data export request"""
    start_time: datetime
    end_time: datetime
    format: str = Field(default="excel", regex="^(excel|csv|json|pdf|html)$")
    vehicle_types: Optional[List[str]] = None


class StatisticsResponse(BaseModel):
    """Statistics response model"""
    total_counts: Dict[str, int]
    counts_by_time: List[Dict[str, Any]]
    avg_speed: Optional[float]
    max_speed: Optional[float]
    peak_hour: Optional[str]
    total_vehicles: int


# Create FastAPI app
app = FastAPI(
    title="LIMA Traffic Counter API",
    description="REST API for traffic counting and analysis",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
app.state.config = None
app.state.db_service = None
app.state.processing_jobs = {}


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    # Load configuration
    app.state.config = AppConfig()

    # Initialize database service
    app.state.db_service = await create_database_service({
        'db_path': str(app.state.config.db_path),
        'host_id': app.state.config.host_id,
        'camera_id': app.state.config.camera_id
    })

    logger.info("API server started")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if app.state.db_service:
        await app.state.db_service.close()

    logger.info("API server shutdown")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "LIMA Traffic Counter API",
        "version": "2.0.0",
        "status": "operational"
    }


@app.post("/counts")
async def add_count(event: CountEvent):
    """Add counting event"""
    try:
        # Create count record
        from src.services.database_service import CountRecord

        record = CountRecord(
            id=None,
            host_id=app.state.config.host_id,
            camera_id=app.state.config.camera_id,
            line_id=event.line_id,
            timestamp=event.timestamp,
            vehicle_type=event.vehicle_type,
            count=event.count,
            direction=event.direction,
            metadata={
                'speed_kmh': event.speed_kmh,
                'confidence': event.confidence
            }
        )

        # Insert to database
        await app.state.db_service.insert_count(record)

        return {"status": "success", "message": "Count recorded"}

    except Exception as e:
        logger.error(f"Failed to add count: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/counts/batch")
async def add_counts_batch(events: List[CountEvent]):
    """Add multiple counting events"""
    try:
        # Convert to count records
        from src.services.database_service import CountRecord

        records = []
        for event in events:
            record = CountRecord(
                id=None,
                host_id=app.state.config.host_id,
                camera_id=app.state.config.camera_id,
                line_id=event.line_id,
                timestamp=event.timestamp,
                vehicle_type=event.vehicle_type,
                count=event.count,
                direction=event.direction,
                metadata={
                    'speed_kmh': event.speed_kmh,
                    'confidence': event.confidence
                }
            )
            records.append(record)

        # Batch insert
        await app.state.db_service.insert_count_batch(records)

        return {
            "status": "success",
            "message": f"{len(records)} counts recorded"
        }

    except Exception as e:
        logger.error(f"Failed to add batch counts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/statistics", response_model=StatisticsResponse)
async def get_statistics(
        start_time: datetime,
        end_time: datetime,
        interval: str = "hour"
):
    """Get counting statistics"""
    try:
        stats = await app.state.db_service.get_statistics(
            start_time, end_time, interval
        )

        return StatisticsResponse(**stats)

    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/export")
async def export_data(request: ExportRequest, background_tasks: BackgroundTasks):
    """Export counting data"""
    try:
        # Get data from database
        df = await app.state.db_service.get_counts(
            request.start_time,
            request.end_time,
            vehicle_types=request.vehicle_types
        )

        if df.empty:
            raise HTTPException(status_code=404, detail="No data found")

        # Generate export file
        export_service = ExportService()

        # Create temporary file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"traffic_export_{timestamp}"

        if request.format == "excel":
            output_path = Path(f"/tmp/{filename}.xlsx")
            await export_service.export_to_excel(df, output_path)
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

        elif request.format == "csv":
            output_path = Path(f"/tmp/{filename}.csv")
            await export_service.export_to_csv(df, output_path)
            media_type = "text/csv"

        elif request.format == "json":
            output_path = Path(f"/tmp/{filename}.json")
            await export_service.export_to_json(df, output_path)
            media_type = "application/json"

        elif request.format == "html":
            output_path = Path(f"/tmp/{filename}.html")
            await export_service.generate_html_dashboard(df, output_path)
            media_type = "text/html"

        else:
            raise HTTPException(status_code=400, detail="Invalid format")

        # Schedule file deletion after download
        background_tasks.add_task(cleanup_file, output_path)

        return FileResponse(
            path=output_path,
            media_type=media_type,
            filename=output_path.name
        )

    except Exception as e:
        logger.error(f"Failed to export data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process")
async def process_video(request: ProcessingRequest, background_tasks: BackgroundTasks):
    """Start video processing job"""
    try:
        # Generate job ID
        job_id = str(uuid.uuid4())

        # Start processing in background
        background_tasks.add_task(
            process_video_task,
            job_id,
            request,
            app.state.config
        )

        # Store job status
        app.state.processing_jobs[job_id] = {
            "status": "processing",
            "progress": 0,
            "start_time": datetime.now()
        }

        return {
            "job_id": job_id,
            "status": "processing",
            "message": "Video processing started"
        }

    except Exception as e:
        logger.error(f"Failed to start processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/process/{job_id}")
async def get_job_status(job_id: str):
    """Get processing job status"""
    if job_id not in app.state.processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return app.state.processing_jobs[job_id]


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload video for processing"""
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            raise HTTPException(status_code=400, detail="Invalid file type")

        # Save uploaded file
        upload_dir = Path("/tmp/uploads")
        upload_dir.mkdir(exist_ok=True)

        file_path = upload_dir / f"{uuid.uuid4()}_{file.filename}"

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        return {
            "filename": file.filename,
            "path": str(file_path),
            "size": len(content)
        }

    except Exception as e:
        logger.error(f"Failed to upload file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()

    try:
        while True:
            # Send periodic updates
            await asyncio.sleep(1)

            # Get latest statistics
            stats = await app.state.db_service.get_statistics(
                datetime.now() - timedelta(minutes=5),
                datetime.now(),
                interval="minute"
            )

            await websocket.send_json({
                "type": "statistics",
                "data": stats
            })

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()


async def process_video_task(job_id: str, request: ProcessingRequest, config: AppConfig):
    """Background task for video processing"""
    try:
        # Update job status
        app.state.processing_jobs[job_id]["status"] = "initializing"

        # Create pipeline configuration
        pipeline_config = {
            'detector': {
                'model_path': str(config.model_dir / f"{request.model}.onnx"),
                'device': request.device,
                'conf_threshold': config.confidence_threshold,
                'nms_threshold': config.nms_threshold
            },
            'tracker': {
                'type': 'bytetrack',
                'track_thresh': config.track_threshold
            },
            'counter': {
                'counting_lines': [
                    {
                        'id': line.id,
                        'start': line.start,
                        'end': line.end,
                        'direction': line.direction,
                        'bidirectional': line.bidirectional
                    }
                    for line in request.counting_lines
                ]
            }
        }

        # Create pipeline
        pipeline = await create_pipeline(pipeline_config)

        # Process video
        # ... (video processing logic)

        # Update job status
        app.state.processing_jobs[job_id]["status"] = "completed"
        app.state.processing_jobs[job_id]["progress"] = 100

    except Exception as e:
        logger.error(f"Processing task error: {e}")
        app.state.processing_jobs[job_id]["status"] = "failed"
        app.state.processing_jobs[job_id]["error"] = str(e)


def cleanup_file(file_path: Path):
    """Clean up temporary file"""
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        logger.error(f"Failed to cleanup file: {e}")


def run_server(config: AppConfig, host: str = "0.0.0.0", port: int = 8000):
    """Run the API server"""
    import uvicorn

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )


def main():
    """Main server entry point"""
    config = AppConfig()
    run_server(config)


if __name__ == "__main__":
    main()