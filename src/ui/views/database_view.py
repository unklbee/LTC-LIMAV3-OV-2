# src/ui/views/database_view.py

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QFileDialog
)
from PySide6.QtCore import Qt, Signal, Slot
from typing import Dict, List
import structlog

logger = structlog.get_logger()


class DatabaseView(QWidget):
    """
    DatabaseView: Simple page for viewing and exporting database records.
    Sync with main UI (sidebar, signals, stats).
    """

    # You can add signals if needed later
    export_requested = Signal(str)  # file format

    def __init__(self):
        super().__init__()
        self.setObjectName("databaseView")
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("Database Records")
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

        # Table for showing records
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Timestamp", "Line", "Vehicle", "Direction", "Speed", "Confidence"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table, 1)

        # Bottom controls
        controls = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._on_refresh)
        controls.addWidget(self.refresh_btn)

        self.export_btn = QPushButton("Export")
        self.export_btn.clicked.connect(self._on_export)
        controls.addWidget(self.export_btn)

        controls.addStretch()
        layout.addLayout(controls)

    @Slot()
    def _on_refresh(self):
        """
        Dummy slot for refresh. Real implementation: fetch from database service.
        """
        logger.info("Refreshing database view...")
        QMessageBox.information(self, "Info", "Refresh not yet implemented.")

    @Slot()
    def _on_export(self):
        """
        Dummy export slot. Real implementation: emit export_requested signal.
        """
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Data", "", "Excel Files (*.xlsx);;CSV Files (*.csv)"
        )
        if file_path:
            QMessageBox.information(self, "Info", f"Exported to: {file_path}")
            # You can emit a signal or call your controller to export data
            # self.export_requested.emit(file_path)

    def update_records(self, records: List[Dict]):
        """
        Call this method from your controller to update table with new records.
        """
        self.table.setRowCount(0)
        for rec in records:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(str(rec.get("timestamp", ""))))
            self.table.setItem(row, 1, QTableWidgetItem(str(rec.get("line", ""))))
            self.table.setItem(row, 2, QTableWidgetItem(str(rec.get("vehicle_type", ""))))
            self.table.setItem(row, 3, QTableWidgetItem(str(rec.get("direction", ""))))
            self.table.setItem(row, 4, QTableWidgetItem(str(rec.get("speed", ""))))
            self.table.setItem(row, 5, QTableWidgetItem(str(rec.get("confidence", ""))))
