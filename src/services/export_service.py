# src/services/export_service.py
"""
Export service for generating reports in various formats.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from io import BytesIO
import json

# Excel support
import xlsxwriter
from openpyxl import Workbook
from openpyxl.chart import LineChart, BarChart, Reference
from openpyxl.utils.dataframe import dataframe_to_rows

# PDF support
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph,
    Spacer, PageBreak, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.barcharts import VerticalBarChart

# Plotting
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

import structlog

logger = structlog.get_logger()


class ExportService:
    """
    Service for exporting traffic data in various formats:
    - Excel with charts
    - CSV
    - PDF reports
    - JSON
    - HTML dashboards
    """

    def __init__(self):
        # Set style for plots
        sns.set_theme(style="darkgrid")
        plt.style.use('dark_background')

    async def export_to_excel(self,
                              data: pd.DataFrame,
                              output_path: Path,
                              include_charts: bool = True,
                              include_summary: bool = True) -> None:
        """Export data to Excel with optional charts and summary"""

        # Run in thread pool to avoid blocking
        await asyncio.to_thread(
            self._export_excel_sync,
            data, output_path, include_charts, include_summary
        )

        logger.info(f"Exported to Excel: {output_path}")

    def _export_excel_sync(self, data: pd.DataFrame, output_path: Path,
                           include_charts: bool, include_summary: bool):
        """Synchronous Excel export"""

        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            workbook = writer.book

            # 1. Raw Data Sheet
            data.to_excel(writer, sheet_name='Raw Data', index=False)
            raw_sheet = writer.sheets['Raw Data']

            # Format headers
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#0078D4',
                'font_color': 'white',
                'border': 1
            })

            for col_num, col_name in enumerate(data.columns):
                raw_sheet.write(0, col_num, col_name, header_format)

            # Auto-fit columns
            for i, col in enumerate(data.columns):
                max_len = max(data[col].astype(str).map(len).max(), len(col)) + 2
                raw_sheet.set_column(i, i, max_len)

            # 2. Summary Sheet
            if include_summary:
                summary_df = self._create_summary(data)
                summary_df.to_excel(writer, sheet_name='Summary', index=True)

                # Format summary
                summary_sheet = writer.sheets['Summary']
                summary_format = workbook.add_format({
                    'num_format': '#,##0',
                    'border': 1
                })

            # 3. Hourly Statistics Sheet
            hourly_df = self._create_hourly_stats(data)
            hourly_df.to_excel(writer, sheet_name='Hourly Stats', index=False)

            # 4. Charts
            if include_charts:
                charts_sheet = workbook.add_worksheet('Charts')

                # Timeline chart
                self._add_timeline_chart(workbook, charts_sheet, hourly_df, 0)

                # Vehicle distribution chart
                self._add_distribution_chart(workbook, charts_sheet, summary_df, 20)

                # Speed histogram
                if 'speed_kmh' in data.columns:
                    self._add_speed_chart(workbook, charts_sheet, data, 40)

    def _create_summary(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create summary statistics"""
        summary = {}

        # Total counts by vehicle type
        if 'vehicle_type' in data.columns:
            vehicle_counts = data.groupby('vehicle_type')['count'].sum()
            for vtype, count in vehicle_counts.items():
                summary[f'Total {vtype}'] = int(count)

        # Overall statistics
        summary['Total Vehicles'] = int(data['count'].sum())
        summary['Time Range'] = f"{data['timestamp'].min()} to {data['timestamp'].max()}"

        # Speed statistics
        if 'speed_kmh' in data.columns:
            speed_data = data[data['speed_kmh'].notna()]['speed_kmh']
            if not speed_data.empty:
                summary['Average Speed'] = f"{speed_data.mean():.1f} km/h"
                summary['Max Speed'] = f"{speed_data.max():.1f} km/h"
                summary['Speed Violations'] = int((speed_data > 60).sum())  # Assuming 60 km/h limit

        # Peak hour analysis
        data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
        hourly_counts = data.groupby('hour')['count'].sum()
        peak_hour = hourly_counts.idxmax()
        summary['Peak Hour'] = f"{peak_hour}:00 - {peak_hour+1}:00"
        summary['Peak Hour Count'] = int(hourly_counts.max())

        return pd.DataFrame(summary.items(), columns=['Metric', 'Value'])

    def _create_hourly_stats(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create hourly statistics"""
        data = data.copy()
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['hour'] = data['timestamp'].dt.floor('H')

        # Aggregate by hour and vehicle type
        hourly = data.groupby(['hour', 'vehicle_type']).agg({
            'count': 'sum',
            'speed_kmh': ['mean', 'max']
        }).reset_index()

        # Flatten column names
        hourly.columns = ['hour', 'vehicle_type', 'count', 'avg_speed', 'max_speed']

        return hourly

    def _add_timeline_chart(self, workbook, worksheet, data, row_start):
        """Add timeline chart to worksheet"""
        # Create chart
        chart = workbook.add_chart({
            'type': 'line',
            'subtype': 'smooth'
        })

        # Configure chart
        chart.set_title({'name': 'Vehicle Count Timeline'})
        chart.set_x_axis({
            'name': 'Time',
            'date_axis': True,
            'major_unit': 1,
            'major_unit_type': 'hours'
        })
        chart.set_y_axis({'name': 'Vehicle Count'})

        # Add series for each vehicle type
        vehicle_types = data['vehicle_type'].unique()
        colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#00BCD4']

        for i, vtype in enumerate(vehicle_types):
            vtype_data = data[data['vehicle_type'] == vtype]

            chart.add_series({
                'name': vtype,
                'categories': ['Hourly Stats', 1, 0, len(vtype_data), 0],
                'values': ['Hourly Stats', 1, 2, len(vtype_data), 2],
                'line': {
                    'color': colors[i % len(colors)],
                    'width': 2
                }
            })

        # Insert chart
        worksheet.insert_chart(row_start, 0, chart, {
            'x_scale': 2,
            'y_scale': 1.5
        })

    def _add_distribution_chart(self, workbook, worksheet, summary, row_start):
        """Add vehicle distribution pie chart"""
        # Create pie chart
        chart = workbook.add_chart({'type': 'pie'})

        # Configure chart
        chart.set_title({'name': 'Vehicle Type Distribution'})

        # Extract vehicle counts from summary
        vehicle_data = summary[summary['Metric'].str.startswith('Total ')].copy()
        vehicle_data = vehicle_data[vehicle_data['Metric'] != 'Total Vehicles']

        if not vehicle_data.empty:
            chart.add_series({
                'name': 'Vehicle Types',
                'categories': ['Summary', 1, 0, len(vehicle_data), 0],
                'values': ['Summary', 1, 1, len(vehicle_data), 1],
                'data_labels': {'percentage': True}
            })

            # Insert chart
            worksheet.insert_chart(row_start, 0, chart, {
                'x_scale': 1.5,
                'y_scale': 1.5
            })

    def _add_speed_chart(self, workbook, worksheet, data, row_start):
        """Add speed distribution histogram"""
        # Create column chart for histogram
        chart = workbook.add_chart({'type': 'column'})

        # Configure chart
        chart.set_title({'name': 'Speed Distribution'})
        chart.set_x_axis({'name': 'Speed (km/h)'})
        chart.set_y_axis({'name': 'Frequency'})

        # Create histogram data
        speed_data = data[data['speed_kmh'].notna()]['speed_kmh']
        hist, bins = np.histogram(speed_data, bins=20)

        # Add series
        chart.add_series({
            'name': 'Speed Distribution',
            'categories': list(bins[:-1]),
            'values': list(hist),
            'fill': {'color': '#2196F3'}
        })

        # Insert chart
        worksheet.insert_chart(row_start, 0, chart, {
            'x_scale': 2,
            'y_scale': 1.5
        })

    async def export_to_csv(self,
                            data: pd.DataFrame,
                            output_path: Path) -> None:
        """Export data to CSV"""
        await asyncio.to_thread(data.to_csv, output_path, index=False)
        logger.info(f"Exported to CSV: {output_path}")

    async def export_to_json(self,
                             data: pd.DataFrame,
                             output_path: Path,
                             orient: str = 'records') -> None:
        """Export data to JSON"""
        await asyncio.to_thread(
            data.to_json, output_path, orient=orient,
            date_format='iso', indent=2
        )
        logger.info(f"Exported to JSON: {output_path}")

    async def generate_pdf_report(self,
                                  data: pd.DataFrame,
                                  output_path: Path,
                                  title: str = "Traffic Analysis Report",
                                  include_charts: bool = True) -> None:
        """Generate PDF report"""
        await asyncio.to_thread(
            self._generate_pdf_sync,
            data, output_path, title, include_charts
        )
        logger.info(f"Generated PDF report: {output_path}")

    def _generate_pdf_sync(self, data: pd.DataFrame, output_path: Path,
                           title: str, include_charts: bool):
        """Synchronous PDF generation"""

        # Create document
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )

        # Container for elements
        elements = []
        styles = getSampleStyleSheet()

        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#0078D4'),
            spaceAfter=30
        )
        elements.append(Paragraph(title, title_style))
        elements.append(Spacer(1, 12))

        # Report metadata
        metadata = [
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Data Range: {data['timestamp'].min()} to {data['timestamp'].max()}",
            f"Total Records: {len(data):,}"
        ]

        for line in metadata:
            elements.append(Paragraph(line, styles['Normal']))
        elements.append(Spacer(1, 20))

        # Executive Summary
        elements.append(Paragraph("Executive Summary", styles['Heading1']))
        summary_df = self._create_summary(data)

        # Convert summary to table
        summary_table_data = [['Metric', 'Value']]
        for _, row in summary_df.iterrows():
            summary_table_data.append([row['Metric'], str(row['Value'])])

        summary_table = Table(summary_table_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0078D4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        elements.append(summary_table)
        elements.append(PageBreak())

        # Charts
        if include_charts:
            elements.append(Paragraph("Visual Analysis", styles['Heading1']))

            # Generate matplotlib charts and add to PDF
            chart_paths = self._generate_charts_for_pdf(data)

            for chart_path, caption in chart_paths:
                img = Image(str(chart_path), width=6*inch, height=4*inch)
                elements.append(img)
                elements.append(Paragraph(caption, styles['Caption']))
                elements.append(Spacer(1, 20))

        # Build PDF
        doc.build(elements)

        # Clean up chart files
        if include_charts:
            for chart_path, _ in chart_paths:
                chart_path.unlink()

    def _generate_charts_for_pdf(self, data: pd.DataFrame) -> List[Tuple[Path, str]]:
        """Generate charts for PDF report"""
        charts = []
        temp_dir = Path("temp_charts")
        temp_dir.mkdir(exist_ok=True)

        # 1. Timeline chart
        plt.figure(figsize=(10, 6))

        data['hour'] = pd.to_datetime(data['timestamp']).dt.floor('H')
        hourly = data.groupby(['hour', 'vehicle_type'])['count'].sum().reset_index()

        for vtype in hourly['vehicle_type'].unique():
            vtype_data = hourly[hourly['vehicle_type'] == vtype]
            plt.plot(vtype_data['hour'], vtype_data['count'],
                     marker='o', label=vtype, linewidth=2)

        plt.xlabel('Time')
        plt.ylabel('Vehicle Count')
        plt.title('Vehicle Count Timeline')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        timeline_path = temp_dir / 'timeline.png'
        plt.savefig(timeline_path, dpi=150, bbox_inches='tight')
        plt.close()

        charts.append((timeline_path, "Figure 1: Vehicle count over time"))

        # 2. Distribution chart
        plt.figure(figsize=(8, 8))

        vehicle_counts = data.groupby('vehicle_type')['count'].sum()
        plt.pie(vehicle_counts.values, labels=vehicle_counts.index,
                autopct='%1.1f%%', startangle=90)
        plt.title('Vehicle Type Distribution')

        dist_path = temp_dir / 'distribution.png'
        plt.savefig(dist_path, dpi=150, bbox_inches='tight')
        plt.close()

        charts.append((dist_path, "Figure 2: Distribution of vehicle types"))

        # 3. Speed distribution (if available)
        if 'speed_kmh' in data.columns:
            plt.figure(figsize=(10, 6))

            speed_data = data[data['speed_kmh'].notna()]['speed_kmh']
            plt.hist(speed_data, bins=30, edgecolor='white', alpha=0.7)
            plt.axvline(speed_data.mean(), color='red', linestyle='--',
                        label=f'Mean: {speed_data.mean():.1f} km/h')
            plt.xlabel('Speed (km/h)')
            plt.ylabel('Frequency')
            plt.title('Speed Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            speed_path = temp_dir / 'speed_dist.png'
            plt.savefig(speed_path, dpi=150, bbox_inches='tight')
            plt.close()

            charts.append((speed_path, "Figure 3: Distribution of vehicle speeds"))

        return charts

    async def generate_html_dashboard(self,
                                      data: pd.DataFrame,
                                      output_path: Path,
                                      title: str = "Traffic Dashboard") -> None:
        """Generate interactive HTML dashboard"""

        # Create HTML template
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #0078D4;
                    margin-bottom: 30px;
                }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin-bottom: 40px;
                }}
                .stat-card {{
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                    border-left: 4px solid #0078D4;
                }}
                .stat-value {{
                    font-size: 32px;
                    font-weight: bold;
                    color: #0078D4;
                    margin: 10px 0;
                }}
                .stat-label {{
                    color: #666;
                    font-size: 14px;
                }}
                .chart-container {{
                    margin-bottom: 40px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{title}</h1>
                
                <div class="stats-grid">
                    {stats_cards}
                </div>
                
                <div class="chart-container" id="timeline-chart"></div>
                <div class="chart-container" id="distribution-chart"></div>
                <div class="chart-container" id="speed-chart"></div>
                
                <script>
                    {plotly_scripts}
                </script>
            </div>
        </body>
        </html>
        """

        # Generate content
        stats_cards = self._generate_stats_cards(data)
        plotly_scripts = await self._generate_plotly_charts(data)

        # Fill template
        html_content = html_template.format(
            title=title,
            stats_cards=stats_cards,
            plotly_scripts=plotly_scripts
        )

        # Write to file
        await asyncio.to_thread(output_path.write_text, html_content)

        logger.info(f"Generated HTML dashboard: {output_path}")

    def _generate_stats_cards(self, data: pd.DataFrame) -> str:
        """Generate HTML stats cards"""
        cards = []

        # Total vehicles
        total = int(data['count'].sum())
        cards.append(f"""
            <div class="stat-card">
                <div class="stat-label">Total Vehicles</div>
                <div class="stat-value">{total:,}</div>
            </div>
        """)

        # Average speed
        if 'speed_kmh' in data.columns:
            avg_speed = data[data['speed_kmh'].notna()]['speed_kmh'].mean()
            cards.append(f"""
                <div class="stat-card">
                    <div class="stat-label">Average Speed</div>
                    <div class="stat-value">{avg_speed:.1f} km/h</div>
                </div>
            """)

        # Time range
        time_range = f"{data['timestamp'].min()} - {data['timestamp'].max()}"
        cards.append(f"""
            <div class="stat-card">
                <div class="stat-label">Time Range</div>
                <div class="stat-value" style="font-size: 16px;">{time_range}</div>
            </div>
        """)

        return '\n'.join(cards)

    async def _generate_plotly_charts(self, data: pd.DataFrame) -> str:
        """Generate Plotly chart scripts"""
        scripts = []

        # Prepare data
        data['hour'] = pd.to_datetime(data['timestamp']).dt.floor('H')
        hourly = data.groupby(['hour', 'vehicle_type'])['count'].sum().reset_index()

        # Timeline chart
        timeline_traces = []
        for vtype in hourly['vehicle_type'].unique():
            vtype_data = hourly[hourly['vehicle_type'] == vtype]
            timeline_traces.append({
                'x': vtype_data['hour'].dt.strftime('%Y-%m-%d %H:%M').tolist(),
                'y': vtype_data['count'].tolist(),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': vtype
            })

        scripts.append(f"""
            var timelineData = {json.dumps(timeline_traces)};
            var timelineLayout = {{
                title: 'Vehicle Count Timeline',
                xaxis: {{ title: 'Time' }},
                yaxis: {{ title: 'Count' }}
            }};
            Plotly.newPlot('timeline-chart', timelineData, timelineLayout);
        """)

        # Distribution chart
        vehicle_counts = data.groupby('vehicle_type')['count'].sum()
        scripts.append(f"""
            var distributionData = [{{
                values: {vehicle_counts.values.tolist()},
                labels: {vehicle_counts.index.tolist()},
                type: 'pie'
            }}];
            var distributionLayout = {{
                title: 'Vehicle Type Distribution'
            }};
            Plotly.newPlot('distribution-chart', distributionData, distributionLayout);
        """)

        return '\n'.join(scripts)