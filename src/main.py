# src/main.py
"""
Main entry point for LIMA Traffic Counter application.
"""

import sys
import asyncio
from pathlib import Path
from typing import Optional

# Qt imports
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from qasync import QEventLoop

# Application imports
from src.ui.controllers.app_controller import AppController
from src.models.config import AppConfig, set_config
from src.utils.logger import setup_logging

import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.CallsiteParameterAdder(
            parameters=[structlog.processors.CallsiteParameter.FILENAME,
                        structlog.processors.CallsiteParameter.LINENO]
        ),
        structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class Application(QApplication):
    """Enhanced QApplication with async support"""

    def __init__(self, argv):
        super().__init__(argv)

        # Application metadata
        self.setApplicationName("LIMA Traffic Counter")
        self.setOrganizationName("Lintas Mediatama")
        self.setOrganizationDomain("lintasmediatama.com")
        self.setApplicationVersion("2.0.0")

        # High DPI support
        # self.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        # self.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

        # Style
        self.setStyle("Fusion")

        # Load configuration
        self.config = self._load_configuration()
        set_config(self.config)

        # Setup logging
        self._setup_logging()

        # Controller
        self.controller: Optional[AppController] = None

    def _load_configuration(self) -> AppConfig:
        """Load application configuration"""
        # Try to load from file first
        config_file = Path.home() / ".lima" / "config.json"

        if config_file.exists():
            try:
                config = AppConfig.load(config_file)
                logger.info("Loaded configuration from file", path=str(config_file))
                return config
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")

        # Fall back to environment/defaults
        config = AppConfig()
        logger.info("Using default/environment configuration")

        return config

    def _setup_logging(self):
        """Setup application logging"""
        log_config = {
            'log_level': self.config.get_log_level_int(),
            'log_to_file': self.config.log_to_file,
            'log_dir': self.config.log_dir,
            'log_max_size': self.config.log_max_size,
            'log_backup_count': self.config.log_backup_count
        }

        setup_logging(**log_config)

    async def initialize(self):
        """Initialize application components"""
        try:
            # Create controller
            self.controller = AppController(self.config)
            await self.controller.initialize()

            # Apply theme
            self._apply_theme()

            # Show main window
            self.controller.show()

            logger.info("Application initialized successfully")

        except Exception as e:
            logger.error(f"Application initialization failed: {e}")
            raise

    def _apply_theme(self):
        """Apply application theme"""
        if self.config.theme == "dark":
            # Dark theme palette
            from PySide6.QtGui import QPalette, QColor

            palette = QPalette()
            palette.setColor(QPalette.Window, QColor(30, 30, 30))
            palette.setColor(QPalette.WindowText, Qt.white)
            palette.setColor(QPalette.Base, QColor(45, 45, 45))
            palette.setColor(QPalette.AlternateBase, QColor(60, 60, 60))
            palette.setColor(QPalette.ToolTipBase, Qt.white)
            palette.setColor(QPalette.ToolTipText, Qt.white)
            palette.setColor(QPalette.Text, Qt.white)
            palette.setColor(QPalette.Button, QColor(45, 45, 45))
            palette.setColor(QPalette.ButtonText, Qt.white)
            palette.setColor(QPalette.BrightText, Qt.red)
            palette.setColor(QPalette.Link, QColor(0, 120, 212))
            palette.setColor(QPalette.Highlight, QColor(0, 120, 212))
            palette.setColor(QPalette.HighlightedText, Qt.black)

            self.setPalette(palette)

            # Additional dark theme styles
            dark_style = """
            QToolTip {
                color: #ffffff;
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
            }
            
            QMenu {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #3d3d3d;
            }
            
            QMenu::item:selected {
                background-color: #0078d4;
            }
            """

            self.setStyleSheet(dark_style)

    async def cleanup(self):
        """Cleanup application resources"""
        logger.info("Cleaning up application...")

        if self.controller:
            await self.controller.cleanup()

        # Save configuration
        config_file = Path.home() / ".lima" / "config.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.config.save(config_file)
            logger.info("Configuration saved", path=str(config_file))
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")

        logger.info("Application cleanup completed")


async def main():
    """Main application entry point"""
    # Create Qt application
    app = Application(sys.argv)

    # Create async event loop
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    try:
        # Initialize application
        await app.initialize()

        # Run event loop
        with loop:
            await loop.run_forever()

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Cleanup
        await app.cleanup()


def run():
    """Run the application"""
    # Handle Ctrl+C gracefully on Windows
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # Check Python version
    if sys.version_info < (3, 10):
        print("Error: Python 3.10 or higher is required")
        sys.exit(1)

    # Run application
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run()


# ==================== Alternative CLI Entry Points ====================

def run_cli():
    """Run command-line interface version"""
    from src.cli.main import main as cli_main
    cli_main()


def run_server():
    """Run API server"""
    from src.server.main import main as server_main
    server_main()


def run_benchmark():
    """Run performance benchmark"""
    from src.utils.benchmark import main as benchmark_main
    benchmark_main()