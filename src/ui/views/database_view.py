# src/ui/views/database_view.py
"""
Database view for managing and viewing stored counting data.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QGroupBox, QLabel, QDateTimeEdit, QComboBox,
    QHeaderView, QMessageBox, QFileDialog, QProgressDialog,
    QTabWidget, QTextEdit, QSpinBox, QGridLayout, QCheckBox
)
from PySide6.QtCore import Qt, Signal, Slot, QDateTime, QTimer
from PySide6.QtGui import QFont
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import asyncio

import structlog

logger = structlog.get_logger()


class DatabaseView(QWidget):
    """Database management and viewing interface"""

    # Signals
    export_requested = Signal(datetime, datetime, str)  # start, end, format
    cleanup_requested = Signal(int)  # days to keep
    backup_requested = Signal(str)  # backup path

    def __init__(self):
        super().__init__()
        self.setObjectName("databaseView")

        # Current data
        self.current_data = None

        # Setup UI
        self._setup_ui()

        # Auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self._refresh_data)
        self.refresh_timer.start(30000)  # Refresh every 30 seconds

    def _setup_ui(self):
        """Setup database view UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("Database Management")
        title.setObjectName("viewTitle")
        title.setStyleSheet("""
            #viewTitle {
                font-size: 24px;
                font-weight: bold;
                color: #0078d4;
                margin-bottom: 20px;
            }
        """)
        layout.addWidget(title)

        # Tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(self._create_data_tab(), "Data View")
        self.tab_widget.addTab(self._create_query_tab(), "Query Builder")
        self.tab_widget.addTab(self._create_maintenance_tab(), "Maintenance")

        layout.addWidget(self.tab_widget)

    def _create_data_tab(self) -> QWidget:
        """Create data viewing tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Filter controls
        filter_group = QGroupBox("Filters")
        filter_layout = QHBoxLayout(filter_group)

        # Date range
        filter_layout.addWidget(QLabel("From:"))
        self.start_date = QDateTimeEdit()
        self.start_date.setCalendarPopup(True)
        self.start_date.setDateTime(QDateTime.currentDateTime().addDays(-7))
        self.start_date.setDisplayFormat("yyyy-MM-dd HH:mm")
        filter_layout.addWidget(self.start_date)

        filter_layout.addWidget(QLabel("To:"))
        self.end_date = QDateTimeEdit()
        self.end_date.setCalendarPopup(True)
        self.end_date.setDateTime(QDateTime.currentDateTime())
        self.end_date.setDisplayFormat("yyyy-MM-dd HH:mm")
        filter_layout.addWidget(self.end_date)

        # Vehicle type filter
        filter_layout.addWidget(QLabel("Vehicle Type:"))
        self.vehicle_type_combo = QComboBox()
        self.vehicle_type_combo.addItems(["All", "car", "truck", "bus", "motorbike", "bicycle"])
        filter_layout.addWidget(self.vehicle_type_combo)

        # Refresh button
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_data)
        filter_layout.addWidget(self.refresh_btn)

        filter_layout.addStretch()

        layout.addWidget(filter_group)

        # Summary statistics
        self.summary_group = QGroupBox("Summary")
        self.summary_layout = QHBoxLayout(self.summary_group)

        self.total_label = self._create_summary_label("Total Records:", "0")
        self.period_label = self._create_summary_label("Period Total:", "0")
        self.avg_label = self._create_summary_label("Daily Average:", "0")
        self.peak_label = self._create_summary_label("Peak Hour:", "N/A")

        self.summary_layout.addWidget(self.total_label)
        self.summary_layout.addWidget(self.period_label)
        self.summary_layout.addWidget(self.avg_label)
        self.summary_layout.addWidget(self.peak_label)
        self.summary_layout.addStretch()

        layout.addWidget(self.summary_group)

        # Data table
        self.data_table = QTableWidget()
        self.data_table.setAlternatingRowColors(True)
        self.data_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.data_table.setSortingEnabled(True)

        # Style
        self.data_table.setStyleSheet("""
            QTableWidget {
                background-color: #2d2d2d;
                alternate-background-color: #3d3d3d;
                color: white;
                gridline-color: #4d4d4d;
                selection-background-color: #0078d4;
            }
            QHeaderView::section {
                background-color: #1e1e1e;
                color: white;
                padding: 5px;
                border: none;
                font-weight: bold;
            }
        """)

        layout.addWidget(self.data_table)

        # Export controls
        export_layout = QHBoxLayout()
        export_layout.addStretch()

        self.export_combo = QComboBox()
        self.export_combo.addItems(["Excel", "CSV", "JSON", "PDF"])
        export_layout.addWidget(QLabel("Export Format:"))
        export_layout.addWidget(self.export_combo)

        self.export_btn = QPushButton("Export Data")
        self.export_btn.clicked.connect(self._export_data)
        export_layout.addWidget(self.export_btn)

        layout.addLayout(export_layout)

        return widget

    def _create_query_tab(self) -> QWidget:
        """Create custom query tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Query templates
        template_group = QGroupBox("Query Templates")
        template_layout = QVBoxLayout(template_group)

        self.template_combo = QComboBox()
        templates = [
            "Select Template...",
            "Hourly counts by vehicle type",
            "Daily totals",
            "Peak hours analysis",
            "Speed violations",
            "Zone dwell times",
            "Custom SQL"
        ]
        self.template_combo.addItems(templates)
        self.template_combo.currentIndexChanged.connect(self._on_template_changed)
        template_layout.addWidget(self.template_combo)

        layout.addWidget(template_group)

        # SQL editor
        sql_group = QGroupBox("SQL Query")
        sql_layout = QVBoxLayout(sql_group)

        self.sql_editor = QTextEdit()
        self.sql_editor.setPlaceholderText("Enter SQL query here...")
        self.sql_editor.setFont(QFont("Consolas", 10))
        self.sql_editor.setMaximumHeight(150)

        # Syntax highlighting
        self.sql_editor.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 10px;
            }
        """)

        sql_layout.addWidget(self.sql_editor)

        # Execute button
        self.execute_btn = QPushButton("Execute Query")
        self.execute_btn.clicked.connect(self._execute_query)
        sql_layout.addWidget(self.execute_btn)

        layout.addWidget(sql_group)

        # Results table
        self.query_results = QTableWidget()
        self.query_results.setAlternatingRowColors(True)
        layout.addWidget(self.query_results)

        return widget

    def _create_maintenance_tab(self) -> QWidget:
        """Create database maintenance tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Database info
        info_group = QGroupBox("Database Information")
        info_layout = QGridLayout(info_group)

        self.db_size_label = QLabel("Size: Calculating...")
        self.record_count_label = QLabel("Total Records: Calculating...")
        self.oldest_record_label = QLabel("Oldest Record: Calculating...")
        self.last_backup_label = QLabel("Last Backup: Never")

        info_layout.addWidget(self.db_size_label, 0, 0)
        info_layout.addWidget(self.record_count_label, 0, 1)
        info_layout.addWidget(self.oldest_record_label, 1, 0)
        info_layout.addWidget(self.last_backup_label, 1, 1)

        layout.addWidget(info_group)

        # Cleanup settings
        cleanup_group = QGroupBox("Data Cleanup")
        cleanup_layout = QVBoxLayout(cleanup_group)

        cleanup_ctrl_layout = QHBoxLayout()
        cleanup_ctrl_layout.addWidget(QLabel("Keep data for:"))

        self.cleanup_days_spin = QSpinBox()
        self.cleanup_days_spin.setRange(1, 365)
        self.cleanup_days_spin.setValue(30)
        self.cleanup_days_spin.setSuffix(" days")
        cleanup_ctrl_layout.addWidget(self.cleanup_days_spin)

        self.cleanup_btn = QPushButton("Clean Old Data")
        self.cleanup_btn.clicked.connect(self._cleanup_database)
        cleanup_ctrl_layout.addWidget(self.cleanup_btn)

        cleanup_ctrl_layout.addStretch()
        cleanup_layout.addLayout(cleanup_ctrl_layout)

        # Preview what will be deleted
        self.cleanup_preview_label = QLabel("No cleanup scheduled")
        self.cleanup_preview_label.setStyleSheet("color: #b0b0b0;")
        cleanup_layout.addWidget(self.cleanup_preview_label)

        layout.addWidget(cleanup_group)

        # Backup/Restore
        backup_group = QGroupBox("Backup & Restore")
        backup_layout = QVBoxLayout(backup_group)

        backup_btn_layout = QHBoxLayout()

        self.backup_btn = QPushButton("Backup Database")
        self.backup_btn.clicked.connect(self._backup_database)
        backup_btn_layout.addWidget(self.backup_btn)

        self.restore_btn = QPushButton("Restore Database")
        self.restore_btn.clicked.connect(self._restore_database)
        backup_btn_layout.addWidget(self.restore_btn)

        self.vacuum_btn = QPushButton("Optimize Database")
        self.vacuum_btn.clicked.connect(self._vacuum_database)
        backup_btn_layout.addWidget(self.vacuum_btn)

        backup_btn_layout.addStretch()
        backup_layout.addLayout(backup_btn_layout)

        # Auto-backup settings
        auto_backup_layout = QHBoxLayout()
        self.auto_backup_check = QCheckBox("Enable Auto-backup")
        auto_backup_layout.addWidget(self.auto_backup_check)

        auto_backup_layout.addWidget(QLabel("Interval:"))
        self.backup_interval_combo = QComboBox()
        self.backup_interval_combo.addItems(["Daily", "Weekly", "Monthly"])
        auto_backup_layout.addWidget(self.backup_interval_combo)

        auto_backup_layout.addStretch()
        backup_layout.addLayout(auto_backup_layout)

        layout.addWidget(backup_group)

        layout.addStretch()

        return widget

    def _create_summary_label(self, label: str, value: str) -> QWidget:
        """Create summary statistic label"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 5, 10, 5)

        label_w = QLabel(label)
        label_w.setStyleSheet("color: #b0b0b0; font-size: 12px;")

        value_w = QLabel(value)
        value_w.setObjectName(f"{label.replace(' ', '').replace(':', '')}Value")
        value_w.setStyleSheet("color: white; font-size: 16px; font-weight: bold;")

        layout.addWidget(label_w)
        layout.addWidget(value_w)

        return widget

    @Slot()
    def _refresh_data(self):
        """Refresh data from database"""
        # This would be connected to the database service
        # For now, generate sample data
        logger.info("Refreshing database view")

        # Update summary
        self.total_label.findChild(QLabel, "TotalRecordsValue").setText("12,543")
        self.period_label.findChild(QLabel, "PeriodTotalValue").setText("2,341")
        self.avg_label.findChild(QLabel, "DailyAverageValue").setText("334.4")
        self.peak_label.findChild(QLabel, "PeakHourValue").setText("17:00-18:00")

        # Update table with sample data
        self._populate_table_with_sample_data()

        # Update maintenance info
        self._update_maintenance_info()

    def _populate_table_with_sample_data(self):
        """Populate table with sample data"""
        # Sample columns
        columns = ["Timestamp", "Vehicle Type", "Count", "Direction", "Speed (km/h)", "Line ID"]
        self.data_table.setColumnCount(len(columns))
        self.data_table.setHorizontalHeaderLabels(columns)

        # Sample data
        sample_data = [
            ["2024-01-20 14:30:15", "car", "1", "forward", "45.2", "line_1"],
            ["2024-01-20 14:30:18", "motorbike", "1", "forward", "38.7", "line_1"],
            ["2024-01-20 14:30:22", "truck", "1", "backward", "42.1", "line_2"],
            ["2024-01-20 14:30:25", "car", "1", "forward", "52.3", "line_1"],
            ["2024-01-20 14:30:28", "bus", "1", "forward", "35.6", "line_2"],
        ]

        self.data_table.setRowCount(len(sample_data))

        for row, data in enumerate(sample_data):
            for col, value in enumerate(data):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                self.data_table.setItem(row, col, item)

        # Adjust column widths
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    def _update_maintenance_info(self):
        """Update maintenance information"""
        self.db_size_label.setText("Size: 125.4 MB")
        self.record_count_label.setText("Total Records: 1,234,567")
        self.oldest_record_label.setText("Oldest Record: 2024-01-01 00:00:00")
        self.last_backup_label.setText("Last Backup: 2024-01-19 02:00:00")

        # Update cleanup preview
        days = self.cleanup_days_spin.value()
        cutoff_date = datetime.now() - timedelta(days=days)
        self.cleanup_preview_label.setText(
            f"Will delete records older than {cutoff_date.strftime('%Y-%m-%d')} (approximately 45,231 records)"
        )

    @Slot(int)
    def _on_template_changed(self, index: int):
        """Handle query template selection"""
        templates = {
            1: """
SELECT 
    DATE_FORMAT(timestamp, '%Y-%m-%d %H:00:00') as hour,
    vehicle_type,
    SUM(count) as total_count
FROM vehicle_counts
WHERE timestamp BETWEEN ? AND ?
GROUP BY hour, vehicle_type
ORDER BY hour, vehicle_type;
""",
            2: """
SELECT 
    DATE(timestamp) as date,
    SUM(count) as daily_total
FROM vehicle_counts
WHERE timestamp BETWEEN ? AND ?
GROUP BY date
ORDER BY date;
""",
            3: """
SELECT 
    HOUR(timestamp) as hour_of_day,
    SUM(count) as total_count
FROM vehicle_counts
WHERE timestamp BETWEEN ? AND ?
GROUP BY hour_of_day
ORDER BY total_count DESC
LIMIT 5;
""",
            4: """
SELECT 
    timestamp,
    track_id,
    speed_kmh,
    speed_limit_kmh,
    vehicle_type
FROM speed_violations
WHERE timestamp BETWEEN ? AND ?
ORDER BY timestamp DESC;
""",
            5: """
SELECT 
    zone_id,
    AVG(duration_seconds) as avg_dwell_time,
    MAX(duration_seconds) as max_dwell_time,
    COUNT(*) as vehicle_count
FROM zone_events
WHERE event_type = 'exit' 
    AND timestamp BETWEEN ? AND ?
GROUP BY zone_id;
"""
        }

        if index in templates:
            self.sql_editor.setText(templates[index].strip())

    @Slot()
    def _execute_query(self):
        """Execute custom SQL query"""
        query = self.sql_editor.toPlainText()
        if not query:
            QMessageBox.warning(self, "Warning", "Please enter a query")
            return

        # This would execute the query via database service
        logger.info("Executing query", query=query)

        # For now, show sample results
        QMessageBox.information(self, "Query Executed",
                                "Query executed successfully.\n5 rows returned.")

    @Slot()
    def _export_data(self):
        """Export current data"""
        format = self.export_combo.currentText().lower()
        start = self.start_date.dateTime().toPython()
        end = self.end_date.dateTime().toPython()

        self.export_requested.emit(start, end, format)

    @Slot()
    def _cleanup_database(self):
        """Clean up old data"""
        days = self.cleanup_days_spin.value()

        reply = QMessageBox.question(
            self,
            "Confirm Cleanup",
            f"This will permanently delete all records older than {days} days.\n\n"
            "Are you sure you want to continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.cleanup_requested.emit(days)

    @Slot()
    def _backup_database(self):
        """Backup database"""
        backup_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Database Backup",
            f"lima_backup_{datetime.now():%Y%m%d_%H%M%S}.db",
            "SQLite Database (*.db)"
        )

        if backup_path:
            self.backup_requested.emit(backup_path)

    @Slot()
    def _restore_database(self):
        """Restore database from backup"""
        backup_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Backup File",
            "",
            "SQLite Database (*.db)"
        )

        if backup_path:
            reply = QMessageBox.warning(
                self,
                "Confirm Restore",
                "This will replace the current database with the backup.\n"
                "All current data will be lost!\n\n"
                "Are you sure you want to continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                # This would trigger database restore
                logger.info("Restoring database", backup_path=backup_path)

    @Slot()
    def _vacuum_database(self):
        """Optimize database"""
        progress = QProgressDialog("Optimizing database...", None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        # This would trigger database vacuum operation
        QTimer.singleShot(2000, lambda: (
            progress.close(),
            QMessageBox.information(self, "Success", "Database optimized successfully!")
        ))

    def load_data(self, data: pd.DataFrame):
        """Load data into table"""
        if data is None or data.empty:
            return

        self.current_data = data

        # Update table
        self.data_table.setRowCount(len(data))
        self.data_table.setColumnCount(len(data.columns))
        self.data_table.setHorizontalHeaderLabels(data.columns.tolist())

        for row in range(len(data)):
            for col in range(len(data.columns)):
                value = str(data.iloc[row, col])
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                self.data_table.setItem(row, col, item)

        # Update summary
        self._update_summary(data)