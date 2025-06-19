# src/cli/main.py
"""
Command-line interface for LIMA Traffic Counter.
"""

import click
import asyncio
import cv2
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

from src.core.pipeline import create_pipeline
from src.models.config import AppConfig
from src.services.database_service import create_database_service
from src.utils.logger import setup_logging

import structlog

logger = structlog.get_logger()


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Path to configuration file')
@click.pass_context
def cli(ctx, config):
    """LIMA Traffic Counter CLI"""
    # Load configuration
    if config:
        ctx.obj = AppConfig.load(Path(config))
    else:
        ctx.obj = AppConfig()

    # Setup logging
    setup_logging(log_level=ctx.obj.get_log_level_int())


@cli.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.option('--model', '-m', default='yolov7-tiny',
              help='Model name to use')
@click.option('--output', '-o', type=click.Path(),
              help='Output video path')
@click.option('--show', '-s', is_flag=True,
              help='Show video during processing')
@click.option('--roi', type=str,
              help='ROI polygon as JSON string')
@click.option('--lines', type=str,
              help='Counting lines as JSON string')
@click.option('--device', '-d', default='AUTO',
              type=click.Choice(['CPU', 'GPU', 'AUTO']),
              help='Inference device')
@click.option('--save-db', is_flag=True,
              help='Save results to database')
@click.pass_obj
def process(config: AppConfig, video_path, model, output, show,
            roi, lines, device, save_db):
    """Process a video file"""

    asyncio.run(_process_video(
        config, video_path, model, output, show,
        roi, lines, device, save_db
    ))


async def _process_video(config: AppConfig, video_path: str, model: str,
                         output: Optional[str], show: bool, roi: Optional[str],
                         lines: Optional[str], device: str, save_db: bool):
    """Process video with counting"""

    # Parse ROI and lines
    roi_polygon = json.loads(roi) if roi else None
    counting_lines = json.loads(lines) if lines else []

    # Create pipeline configuration
    pipeline_config = {
        'detector': {
            'model_path': str(config.model_dir / f"{model}.onnx"),
            'device': device,
            'conf_threshold': config.confidence_threshold,
            'nms_threshold': config.nms_threshold
        },
        'tracker': {
            'type': 'bytetrack',
            'track_thresh': config.track_threshold,
            'track_buffer': config.track_buffer,
            'max_age': config.max_age
        },
        'counter': {
            'counting_lines': counting_lines,
            'enable_speed_estimation': config.enable_speed_estimation,
            'pixel_per_meter': config.pixel_per_meter
        },
        'batch_size': config.batch_size
    }

    # Create pipeline
    pipeline = await create_pipeline(pipeline_config)

    # Create database service if needed
    db_service = None
    if save_db:
        db_service = await create_database_service({
            'db_path': str(config.db_path),
            'host_id': config.host_id,
            'camera_id': config.camera_id
        })

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create output writer if needed
    writer = None
    if output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output, fourcc, fps, (width, height))

    # Progress bar
    with click.progressbar(length=total_frames, label='Processing') as pbar:
        frame_count = 0

        # Process frames
        async for frame_data in pipeline.start(cap):
            frame = frame_data.raw_frame

            # Draw detections
            if frame_data.tracks:
                for track in frame_data.tracks:
                    x1, y1, x2, y2 = track.bbox.astype(int)

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw track ID
                    label = f"ID: {track.track_id} ({track.class_id})"
                    cv2.putText(frame, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1)

            # Draw counting lines
            for line in counting_lines:
                cv2.line(frame,
                         tuple(map(int, line['start'])),
                         tuple(map(int, line['end'])),
                         (0, 0, 255), 2)

            # Show frame if requested
            if show:
                cv2.imshow('LIMA Traffic Counter', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Write output frame
            if writer:
                writer.write(frame)

            # Save to database
            if db_service and frame_data.metadata.get('counts'):
                # Save counts to database
                pass

            frame_count += 1
            pbar.update(1)

    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # Get final statistics
    stats = pipeline.get_stats()
    counter_stats = pipeline.counter.get_statistics()

    # Print results
    click.echo("\n=== Processing Complete ===")
    click.echo(f"Frames processed: {stats.frames_processed}")
    click.echo(f"Average FPS: {stats.avg_fps:.1f}")
    click.echo(f"Total counts: {counter_stats['total_counts']}")

    # Close services
    await pipeline.stop()
    if db_service:
        await db_service.close()


@cli.command()
@click.option('--start', '-s', type=click.DateTime(),
              default=str(datetime.now().replace(hour=0, minute=0, second=0)),
              help='Start datetime')
@click.option('--end', '-e', type=click.DateTime(),
              default=str(datetime.now()),
              help='End datetime')
@click.option('--format', '-f',
              type=click.Choice(['excel', 'csv', 'json', 'html']),
              default='excel', help='Export format')
@click.option('--output', '-o', type=click.Path(), required=True,
              help='Output file path')
@click.pass_obj
def export(config: AppConfig, start, end, format, output):
    """Export counting data"""

    asyncio.run(_export_data(config, start, end, format, output))


async def _export_data(config: AppConfig, start: datetime, end: datetime,
                       format: str, output: str):
    """Export data from database"""

    # Create database service
    db_service = await create_database_service({
        'db_path': str(config.db_path),
        'host_id': config.host_id,
        'camera_id': config.camera_id
    })

    # Get data
    df = await db_service.get_counts(start, end)

    if df.empty:
        click.echo("No data found for specified time range")
        return

    # Export based on format
    from src.services.export_service import ExportService
    export_service = ExportService()

    output_path = Path(output)

    if format == 'excel':
        await export_service.export_to_excel(df, output_path,
                                             include_charts=True)
    elif format == 'csv':
        await export_service.export_to_csv(df, output_path)
    elif format == 'json':
        await export_service.export_to_json(df, output_path)
    elif format == 'html':
        await export_service.generate_html_dashboard(df, output_path)

    click.echo(f"Data exported to: {output_path}")

    # Get statistics
    stats = await db_service.get_statistics(start, end)

    click.echo("\n=== Export Summary ===")
    click.echo(f"Total vehicles: {stats['total_vehicles']}")
    click.echo(f"Vehicle breakdown: {stats['total_counts']}")
    if stats['avg_speed']:
        click.echo(f"Average speed: {stats['avg_speed']:.1f} km/h")

    await db_service.close()


@cli.command()
@click.option('--model', '-m', default='yolov7-tiny',
              help='Model to benchmark')
@click.option('--video', '-v', type=click.Path(exists=True),
              help='Test video path')
@click.option('--duration', '-d', default=60,
              help='Test duration in seconds')
@click.option('--output', '-o', type=click.Path(),
              default='benchmark_results',
              help='Output directory')
@click.pass_obj
def benchmark(config: AppConfig, model, video, duration, output):
    """Run performance benchmark"""

    from src.utils.benchmark import BenchmarkConfig, PerformanceBenchmark

    # Create benchmark configuration
    bench_config = BenchmarkConfig()
    bench_config.models = [model]
    bench_config.video_path = video
    bench_config.duration_seconds = duration
    bench_config.output_dir = Path(output)

    # Run benchmark
    benchmark = PerformanceBenchmark(bench_config)
    asyncio.run(benchmark.run())

    click.echo(f"Benchmark results saved to: {output}")


@cli.command()
@click.option('--days', '-d', default=30,
              help='Keep data for N days')
@click.option('--vacuum', is_flag=True,
              help='Vacuum database after cleanup')
@click.pass_obj
def cleanup(config: AppConfig, days, vacuum):
    """Clean up old data from database"""

    asyncio.run(_cleanup_database(config, days, vacuum))


async def _cleanup_database(config: AppConfig, days: int, vacuum: bool):
    """Clean up old database records"""

    # Create database service
    db_service = await create_database_service({
        'db_path': str(config.db_path),
        'host_id': config.host_id,
        'camera_id': config.camera_id
    })

    # Get current statistics before cleanup
    from datetime import timedelta
    cutoff_date = datetime.now() - timedelta(days=days)

    click.echo(f"Cleaning records older than {cutoff_date.strftime('%Y-%m-%d')}")

    # Perform cleanup
    await db_service.cleanup_old_data(days)

    # Vacuum if requested
    if vacuum:
        click.echo("Vacuuming database...")
        await db_service.vacuum()

    click.echo("Cleanup completed successfully")

    await db_service.close()


@cli.command()
@click.option('--host', '-h', default='0.0.0.0',
              help='Server host')
@click.option('--port', '-p', default=8000,
              help='Server port')
@click.pass_obj
def serve(config: AppConfig, host, port):
    """Start API server"""

    click.echo(f"Starting API server on {host}:{port}")

    # Import and run server
    from src.server.main import run_server
    run_server(config, host, port)


def main():
    """Main CLI entry point"""
    cli(obj=None)


if __name__ == '__main__':
    main()