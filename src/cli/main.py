# src/cli/main.py

"""
LIMA Traffic Counter - Command Line Interface

This module provides a comprehensive CLI for the LIMA Traffic Counter system,
including detection, database management, and system utilities.
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

from ..object_detector.config import Config
from ..object_detector.cli import main as detection_main
from ..services.database_service import create_database_service

logger = logging.getLogger(__name__)

def setup_logging(verbose: bool = False):
    """Setup logging for CLI operations."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def cmd_detect(args):
    """Run object detection (delegates to existing detection CLI)."""
    print("Starting LIMA Traffic Detection...")

    # Prepare arguments for the detection CLI
    detection_args = []

    if hasattr(args, 'source') and args.source:
        detection_args.extend(['--source', str(args.source)])
    if hasattr(args, 'weights') and args.weights:
        detection_args.extend(['--weights', str(args.weights)])
    if hasattr(args, 'device') and args.device:
        detection_args.extend(['--device', args.device])
    if hasattr(args, 'conf_thres') and args.conf_thres:
        detection_args.extend(['--conf-thres', str(args.conf_thres)])
    if hasattr(args, 'names') and args.names:
        detection_args.extend(['--names', str(args.names)])

    # Override sys.argv temporarily to pass args to detection main
    original_argv = sys.argv[:]
    sys.argv = ['lima-detect'] + detection_args

    try:
        detection_main()
    finally:
        sys.argv = original_argv

def cmd_gui(args):
    """Launch the GUI application."""
    print("Launching LIMA Traffic Counter GUI...")
    from ..object_detector.gui.main_window import main as gui_main
    gui_main()

def cmd_server(args):
    """Start the API server."""
    print("Starting LIMA API Server...")
    from ..server.main import main as server_main

    # Add server-specific arguments to sys.argv if needed
    if args.debug:
        sys.argv.append('--debug')

    server_main()

def cmd_db_info(args):
    """Show database information."""
    try:
        with create_database_service() as db_service:
            info = db_service.get_connection_info()
            print("Database Information:")
            print(f"  Path: {info['db_path']}")
            print(f"  Host ID: {info['host_id']}")
            print(f"  API URL: {info['api_url']}")
            print(f"  Status: {info['status']}")

            if db_service.is_available():
                db_path = Path(info['db_path'])
                if db_path.exists():
                    size = db_path.stat().st_size
                    print(f"  File size: {size} bytes ({size/1024:.1f} KB)")
                    print(f"  Last modified: {datetime.fromtimestamp(db_path.stat().st_mtime)}")
    except Exception as e:
        print(f"Error accessing database: {e}")
        return 1
    return 0

def cmd_db_test(args):
    """Test database connection and operations."""
    print("Testing database connection...")

    try:
        with create_database_service() as db_service:
            if not db_service.is_available():
                print("❌ Database connection failed")
                return 1

            print("✅ Database connection successful")

            # Test saving a dummy record
            from ..services.database_service import CountRecord
            test_record = CountRecord(
                host_id="test-cli",
                interval_start=datetime.now() - timedelta(minutes=1),
                interval_end=datetime.now(),
                car=5,
                bicycle=2,
                truck=1
            )

            if db_service.save_count_record(test_record):
                print("✅ Test record saved successfully")
            else:
                print("❌ Failed to save test record")
                return 1

    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return 1

    print("✅ All database tests passed")
    return 0

def cmd_config(args):
    """Show or modify configuration."""
    if args.show:
        print("LIMA Traffic Counter Configuration:")
        print(f"  Project Root: {Config.PROJECT_ROOT}")
        print(f"  Model Directory: {Config.MODEL_DIR}")
        print(f"  Data Directory: {Config.DATA_DIR}")
        print(f"  Weights Path: {Config.WEIGHTS_PATH}")
        print(f"  Names Path: {Config.NAMES_PATH}")
        print(f"  Database Path: {Config.DB_PATH}")
        print(f"  Input Size: {Config.INPUT_SIZE}")
        print(f"  Confidence Threshold: {Config.CONFIDENCE_THRESHOLD}")
        print(f"  Save Interval: {Config.COUNT_SAVE_INTERVAL_SEC} seconds")
        print(f"  Host ID: {Config.HOST_ID}")
        print(f"  API URL: {Config.API_URL}")
        print(f"  Vehicle Classes: {', '.join(Config.VEHICLE_CLASSES)}")
        print(f"  Log Level: {Config.LOG_LEVEL}")

        # Check if paths exist
        print("\nPath Status:")
        for name, path in [
            ("Model Dir", Config.MODEL_DIR),
            ("Data Dir", Config.DATA_DIR),
            ("Weights", Config.WEIGHTS_PATH),
            ("Names", Config.NAMES_PATH),
            ("Database", Config.DB_PATH)
        ]:
            status = "✅ exists" if Path(path).exists() else "❌ missing"
            print(f"  {name}: {status}")

def cmd_version(args):
    """Show version information."""
    print("LIMA Traffic Counter")
    print("Version: 1.0.0")
    print("OpenVINO Edition")
    print()

    # Try to get OpenVINO version
    try:
        import openvino as ov
        print(f"OpenVINO Version: {ov.__version__}")
    except ImportError:
        print("OpenVINO: Not installed")

    # Other dependencies
    try:
        import cv2
        print(f"OpenCV Version: {cv2.__version__}")
    except ImportError:
        print("OpenCV: Not installed")

    try:
        from PySide6 import __version__
        print(f"PySide6 Version: {__version__}")
    except ImportError:
        print("PySide6: Not installed")

def create_parser():
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog='lima',
        description='LIMA Traffic Counter - Comprehensive CLI'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands'
    )

    # Detection command
    detect_parser = subparsers.add_parser(
        'detect',
        help='Run traffic detection on video source'
    )
    detect_parser.add_argument(
        '--source',
        default='0',
        help='Video source (webcam=0, file path, or RTSP URL)'
    )
    detect_parser.add_argument(
        '--weights',
        default=str(Config.WEIGHTS_PATH),
        help='Path to model weights (.xml or .onnx)'
    )
    detect_parser.add_argument(
        '--device',
        default='AUTO',
        choices=['CPU', 'GPU', 'AUTO', 'NPU'],
        help='OpenVINO device'
    )
    detect_parser.add_argument(
        '--conf-thres',
        type=float,
        default=Config.CONFIDENCE_THRESHOLD,
        help='Confidence threshold'
    )
    detect_parser.add_argument(
        '--names',
        default=str(Config.NAMES_PATH),
        help='Path to class names file'
    )
    detect_parser.set_defaults(func=cmd_detect)

    # GUI command
    gui_parser = subparsers.add_parser(
        'gui',
        help='Launch the graphical user interface'
    )
    gui_parser.set_defaults(func=cmd_gui)

    # Server command
    server_parser = subparsers.add_parser(
        'server',
        help='Start the API server'
    )
    server_parser.add_argument(
        '--debug',
        action='store_true',
        help='Run server in debug mode with auto-reload'
    )
    server_parser.set_defaults(func=cmd_server)

    # Database commands
    db_parser = subparsers.add_parser(
        'db',
        help='Database management commands'
    )
    db_subparsers = db_parser.add_subparsers(dest='db_command')

    db_info_parser = db_subparsers.add_parser(
        'info',
        help='Show database information'
    )
    db_info_parser.set_defaults(func=cmd_db_info)

    db_test_parser = db_subparsers.add_parser(
        'test',
        help='Test database connection'
    )
    db_test_parser.set_defaults(func=cmd_db_test)

    # Config command
    config_parser = subparsers.add_parser(
        'config',
        help='Configuration management'
    )
    config_parser.add_argument(
        '--show',
        action='store_true',
        help='Show current configuration'
    )
    config_parser.set_defaults(func=cmd_config)

    # Version command
    version_parser = subparsers.add_parser(
        'version',
        help='Show version information'
    )
    version_parser.set_defaults(func=cmd_version)

    return parser

def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Ensure directories exist
    Config.ensure_dirs()

    # Handle case where no command is provided
    if not hasattr(args, 'func'):
        if args.command == 'db' and not args.db_command:
            parser.parse_args(['db', '--help'])
        else:
            parser.print_help()
            return 1

    # Execute the command
    try:
        result = args.func(args)
        return result if result is not None else 0
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Command failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())