# src/utils/benchmark.py
"""
Performance benchmark utility for testing different configurations.
"""

import asyncio
import time
import psutil
import GPUtil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import cv2
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns

from src.core.pipeline import create_pipeline
from src.core.detector import create_detector
from src.core.tracker import create_tracker
from src.core.counter import create_counter
from src.models.config import AppConfig
from src.utils.logger import PerformanceLogger, Timer

import structlog

logger = structlog.get_logger()


class BenchmarkConfig:
    """Benchmark configuration"""

    def __init__(self):
        self.video_path: Optional[str] = None
        self.duration_seconds: int = 60
        self.models: List[str] = ["yolov7-tiny", "yolov7", "yolov8n", "yolov8s"]
        self.devices: List[str] = ["CPU", "GPU", "AUTO"]
        self.batch_sizes: List[int] = [1, 2, 4, 8]
        self.input_sizes: List[Tuple[int, int]] = [(640, 640), (480, 480), (320, 320)]
        self.output_dir: Path = Path("benchmark_results")
        self.save_video: bool = False
        self.warmup_frames: int = 50


class BenchmarkResult:
    """Container for benchmark results"""

    def __init__(self, config_name: str):
        self.config_name = config_name
        self.metrics = {
            'fps': [],
            'inference_time_ms': [],
            'cpu_usage': [],
            'memory_usage_mb': [],
            'gpu_usage': [],
            'gpu_memory_mb': [],
            'accuracy': [],
            'frames_processed': 0,
            'frames_dropped': 0,
            'total_time': 0.0
        }

    def add_frame_metrics(self, fps: float, inference_ms: float,
                          cpu_percent: float, memory_mb: float,
                          gpu_percent: Optional[float] = None,
                          gpu_memory_mb: Optional[float] = None):
        """Add metrics for a single frame"""
        self.metrics['fps'].append(fps)
        self.metrics['inference_time_ms'].append(inference_ms)
        self.metrics['cpu_usage'].append(cpu_percent)
        self.metrics['memory_usage_mb'].append(memory_mb)

        if gpu_percent is not None:
            self.metrics['gpu_usage'].append(gpu_percent)
        if gpu_memory_mb is not None:
            self.metrics['gpu_memory_mb'].append(gpu_memory_mb)

        self.metrics['frames_processed'] += 1

    def get_summary(self) -> Dict:
        """Get summary statistics"""
        summary = {
            'config': self.config_name,
            'frames_processed': self.metrics['frames_processed'],
            'frames_dropped': self.metrics['frames_dropped'],
            'total_time': self.metrics['total_time']
        }

        # Calculate statistics for each metric
        for metric, values in self.metrics.items():
            if isinstance(values, list) and values:
                summary[f'{metric}_mean'] = np.mean(values)
                summary[f'{metric}_std'] = np.std(values)
                summary[f'{metric}_min'] = np.min(values)
                summary[f'{metric}_max'] = np.max(values)
                summary[f'{metric}_p95'] = np.percentile(values, 95)

        return summary


class PerformanceBenchmark:
    """Main benchmark runner"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[BenchmarkResult] = []
        self.perf_logger = PerformanceLogger("benchmark")

        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    async def run(self):
        """Run complete benchmark suite"""
        logger.info("Starting performance benchmark...")

        # Get test configurations
        test_configs = self._generate_test_configs()

        # Run each configuration
        for i, test_config in enumerate(test_configs):
            logger.info(f"Running test {i+1}/{len(test_configs)}: {test_config['name']}")

            try:
                result = await self._run_single_test(test_config)
                self.results.append(result)

                # Save intermediate results
                self._save_results()

            except Exception as e:
                logger.error(f"Test failed: {e}")
                continue

        # Generate report
        self._generate_report()

        logger.info("Benchmark completed!")

    def _generate_test_configs(self) -> List[Dict]:
        """Generate test configurations"""
        configs = []

        # Test different models
        for model in self.config.models:
            for device in self.config.devices:
                for batch_size in self.config.batch_sizes:
                    # Skip invalid combinations
                    if device == "GPU" and not self._check_gpu_available():
                        continue

                    config = {
                        'name': f"{model}_{device}_batch{batch_size}",
                        'model': model,
                        'device': device,
                        'batch_size': batch_size,
                        'input_size': (640, 640)  # Default
                    }
                    configs.append(config)

        # Test different input sizes
        for input_size in self.config.input_sizes:
            config = {
                'name': f"default_{input_size[0]}x{input_size[1]}",
                'model': self.config.models[0],
                'device': 'AUTO',
                'batch_size': 4,
                'input_size': input_size
            }
            configs.append(config)

        return configs

    async def _run_single_test(self, test_config: Dict) -> BenchmarkResult:
        """Run a single benchmark test"""
        result = BenchmarkResult(test_config['name'])

        # Create pipeline
        pipeline_config = {
            'detector': {
                'model_path': self._get_model_path(test_config['model']),
                'device': test_config['device'],
                'conf_threshold': 0.25,
                'nms_threshold': 0.45
            },
            'tracker': {
                'type': 'bytetrack',
                'track_thresh': 0.5
            },
            'counter': {
                'counting_lines': []  # No counting for benchmark
            },
            'batch_size': test_config['batch_size'],
            'use_gpu_decode': test_config['device'] == 'GPU'
        }

        pipeline = await create_pipeline(pipeline_config)

        # Open video
        if self.config.video_path:
            video_source = cv2.VideoCapture(self.config.video_path)
        else:
            # Generate synthetic video
            video_source = self._create_synthetic_video(test_config['input_size'])

        # Warmup
        logger.info(f"Warming up for {self.config.warmup_frames} frames...")
        await self._warmup(pipeline, video_source)

        # Reset video
        video_source.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Run benchmark
        start_time = time.time()
        frame_count = 0
        last_frame_time = start_time

        # System monitor
        process = psutil.Process()

        async for frame_data in pipeline.start(video_source):
            current_time = time.time()

            # Calculate metrics
            fps = 1.0 / (current_time - last_frame_time)
            last_frame_time = current_time

            # Get pipeline statistics
            pipeline_stats = pipeline.get_stats()
            inference_time = pipeline_stats.stage_timings.get('inference', 0) * 1000

            # System metrics
            cpu_percent = process.cpu_percent()
            memory_mb = process.memory_info().rss / 1024 / 1024

            # GPU metrics (if available)
            gpu_percent = None
            gpu_memory_mb = None

            if test_config['device'] == 'GPU':
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        gpu_percent = gpu.load * 100
                        gpu_memory_mb = gpu.memoryUsed
                except:
                    pass

            # Add metrics
            result.add_frame_metrics(
                fps=fps,
                inference_ms=inference_time,
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                gpu_percent=gpu_percent,
                gpu_memory_mb=gpu_memory_mb
            )

            # Log performance
            self.perf_logger.log_fps(fps)
            self.perf_logger.log_inference_time(inference_time, test_config['batch_size'])
            self.perf_logger.log_memory_usage(memory_mb, gpu_memory_mb)

            frame_count += 1

            # Check duration
            if current_time - start_time >= self.config.duration_seconds:
                break

        # Calculate total metrics
        result.metrics['total_time'] = time.time() - start_time
        result.metrics['frames_dropped'] = pipeline_stats.frames_dropped

        # Cleanup
        await pipeline.stop()
        video_source.release()

        return result

    async def _warmup(self, pipeline, video_source):
        """Warmup pipeline"""
        frame_count = 0

        async for frame_data in pipeline.start(video_source):
            frame_count += 1
            if frame_count >= self.config.warmup_frames:
                break

    def _create_synthetic_video(self, size: Tuple[int, int]):
        """Create synthetic video for testing"""
        class SyntheticVideo:
            def __init__(self, size):
                self.size = size
                self.frame_count = 0

            def read(self, buffer=None):
                # Generate random frame with some objects
                frame = np.random.randint(0, 255, (*self.size[::-1], 3), dtype=np.uint8)

                # Add some rectangles to simulate objects
                for _ in range(np.random.randint(5, 15)):
                    x1 = np.random.randint(0, self.size[0] - 50)
                    y1 = np.random.randint(0, self.size[1] - 50)
                    x2 = x1 + np.random.randint(20, 50)
                    y2 = y1 + np.random.randint(20, 50)
                    color = tuple(np.random.randint(0, 255, 3).tolist())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)

                self.frame_count += 1
                return True, frame

            def set(self, prop, value):
                if prop == cv2.CAP_PROP_POS_FRAMES:
                    self.frame_count = value

            def release(self):
                pass

        return SyntheticVideo(size)

    def _get_model_path(self, model_name: str) -> str:
        """Get model file path"""
        model_dir = Path("models")

        # Try different extensions
        for ext in ['.xml', '.onnx', '.engine']:
            path = model_dir / f"{model_name}{ext}"
            if path.exists():
                return str(path)

        # Default
        return str(model_dir / f"{model_name}.onnx")

    def _check_gpu_available(self) -> bool:
        """Check if GPU is available"""
        try:
            gpus = GPUtil.getGPUs()
            return len(gpus) > 0
        except:
            return False

    def _save_results(self):
        """Save intermediate results"""
        # Convert results to DataFrame
        data = []
        for result in self.results:
            summary = result.get_summary()
            data.append(summary)

        df = pd.DataFrame(data)

        # Save as CSV
        csv_path = self.config.output_dir / "benchmark_results.csv"
        df.to_csv(csv_path, index=False)

        # Save as JSON
        json_path = self.config.output_dir / "benchmark_results.json"
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _generate_report(self):
        """Generate comprehensive benchmark report"""
        # Create report directory
        report_dir = self.config.output_dir / f"report_{datetime.now():%Y%m%d_%H%M%S}"
        report_dir.mkdir(parents=True, exist_ok=True)

        # Convert results to DataFrame
        data = []
        for result in self.results:
            summary = result.get_summary()
            data.append(summary)

        df = pd.DataFrame(data)

        # Generate plots
        self._plot_fps_comparison(df, report_dir)
        self._plot_inference_time(df, report_dir)
        self._plot_resource_usage(df, report_dir)
        self._plot_detailed_metrics(report_dir)

        # Generate HTML report
        self._generate_html_report(df, report_dir)

        logger.info(f"Report generated at: {report_dir}")

    def _plot_fps_comparison(self, df: pd.DataFrame, output_dir: Path):
        """Plot FPS comparison"""
        plt.figure(figsize=(12, 6))

        # Bar plot
        configs = df['config']
        fps_mean = df['fps_mean']
        fps_std = df['fps_std']

        x = np.arange(len(configs))
        plt.bar(x, fps_mean, yerr=fps_std, capsize=5)
        plt.xticks(x, configs, rotation=45, ha='right')
        plt.xlabel('Configuration')
        plt.ylabel('FPS')
        plt.title('FPS Comparison Across Configurations')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'fps_comparison.png', dpi=150)
        plt.close()

    def _plot_inference_time(self, df: pd.DataFrame, output_dir: Path):
        """Plot inference time comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Mean inference time
        ax1.bar(range(len(df)), df['inference_time_ms_mean'])
        ax1.set_xticks(range(len(df)))
        ax1.set_xticklabels(df['config'], rotation=45, ha='right')
        ax1.set_ylabel('Inference Time (ms)')
        ax1.set_title('Mean Inference Time')
        ax1.grid(True, alpha=0.3)

        # 95th percentile
        ax2.bar(range(len(df)), df['inference_time_ms_p95'])
        ax2.set_xticks(range(len(df)))
        ax2.set_xticklabels(df['config'], rotation=45, ha='right')
        ax2.set_ylabel('Inference Time (ms)')
        ax2.set_title('95th Percentile Inference Time')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'inference_time.png', dpi=150)
        plt.close()

    def _plot_resource_usage(self, df: pd.DataFrame, output_dir: Path):
        """Plot resource usage"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # CPU usage
        axes[0, 0].bar(range(len(df)), df['cpu_usage_mean'])
        axes[0, 0].set_xticks(range(len(df)))
        axes[0, 0].set_xticklabels(df['config'], rotation=45, ha='right')
        axes[0, 0].set_ylabel('CPU Usage (%)')
        axes[0, 0].set_title('Mean CPU Usage')
        axes[0, 0].grid(True, alpha=0.3)

        # Memory usage
        axes[0, 1].bar(range(len(df)), df['memory_usage_mb_mean'])
        axes[0, 1].set_xticks(range(len(df)))
        axes[0, 1].set_xticklabels(df['config'], rotation=45, ha='right')
        axes[0, 1].set_ylabel('Memory Usage (MB)')
        axes[0, 1].set_title('Mean Memory Usage')
        axes[0, 1].grid(True, alpha=0.3)

        # GPU usage (if available)
        if 'gpu_usage_mean' in df.columns:
            gpu_configs = df[df['gpu_usage_mean'].notna()]
            if not gpu_configs.empty:
                axes[1, 0].bar(range(len(gpu_configs)), gpu_configs['gpu_usage_mean'])
                axes[1, 0].set_xticks(range(len(gpu_configs)))
                axes[1, 0].set_xticklabels(gpu_configs['config'], rotation=45, ha='right')
                axes[1, 0].set_ylabel('GPU Usage (%)')
                axes[1, 0].set_title('Mean GPU Usage')
                axes[1, 0].grid(True, alpha=0.3)

        # GPU memory (if available)
        if 'gpu_memory_mb_mean' in df.columns:
            gpu_configs = df[df['gpu_memory_mb_mean'].notna()]
            if not gpu_configs.empty:
                axes[1, 1].bar(range(len(gpu_configs)), gpu_configs['gpu_memory_mb_mean'])
                axes[1, 1].set_xticks(range(len(gpu_configs)))
                axes[1, 1].set_xticklabels(gpu_configs['config'], rotation=45, ha='right')
                axes[1, 1].set_ylabel('GPU Memory (MB)')
                axes[1, 1].set_title('Mean GPU Memory Usage')
                axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'resource_usage.png', dpi=150)
        plt.close()

    def _plot_detailed_metrics(self, output_dir: Path):
        """Plot detailed metrics over time"""
        # Create figure for each configuration
        for result in self.results[:3]:  # Limit to first 3 for brevity
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Detailed Metrics: {result.config_name}')

            # FPS over time
            axes[0, 0].plot(result.metrics['fps'])
            axes[0, 0].set_xlabel('Frame')
            axes[0, 0].set_ylabel('FPS')
            axes[0, 0].set_title('FPS Over Time')
            axes[0, 0].grid(True, alpha=0.3)

            # Inference time over time
            axes[0, 1].plot(result.metrics['inference_time_ms'])
            axes[0, 1].set_xlabel('Frame')
            axes[0, 1].set_ylabel('Inference Time (ms)')
            axes[0, 1].set_title('Inference Time Over Time')
            axes[0, 1].grid(True, alpha=0.3)

            # CPU usage over time
            axes[1, 0].plot(result.metrics['cpu_usage'])
            axes[1, 0].set_xlabel('Frame')
            axes[1, 0].set_ylabel('CPU Usage (%)')
            axes[1, 0].set_title('CPU Usage Over Time')
            axes[1, 0].grid(True, alpha=0.3)

            # Memory usage over time
            axes[1, 1].plot(result.metrics['memory_usage_mb'])
            axes[1, 1].set_xlabel('Frame')
            axes[1, 1].set_ylabel('Memory Usage (MB)')
            axes[1, 1].set_title('Memory Usage Over Time')
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / f'detailed_{result.config_name}.png', dpi=150)
            plt.close()

    def _generate_html_report(self, df: pd.DataFrame, output_dir: Path):
        """Generate HTML report"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>LIMA Traffic Counter - Performance Benchmark Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #0078D4;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #0078D4;
                    color: white;
                }}
                tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                img {{
                    max-width: 100%;
                    margin: 20px 0;
                }}
                .metric {{
                    display: inline-block;
                    margin: 10px;
                    padding: 15px;
                    background-color: #f0f0f0;
                    border-radius: 5px;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #0078D4;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Performance Benchmark Report</h1>
                <p>Generated: {timestamp}</p>
                
                <h2>Summary</h2>
                <div>
                    <div class="metric">
                        <div>Best FPS</div>
                        <div class="metric-value">{best_fps:.1f}</div>
                        <div>{best_fps_config}</div>
                    </div>
                    <div class="metric">
                        <div>Lowest Latency</div>
                        <div class="metric-value">{best_latency:.1f} ms</div>
                        <div>{best_latency_config}</div>
                    </div>
                    <div class="metric">
                        <div>Most Efficient</div>
                        <div class="metric-value">{best_efficiency:.1f} FPS/W</div>
                        <div>{best_efficiency_config}</div>
                    </div>
                </div>
                
                <h2>Results Table</h2>
                {results_table}
                
                <h2>Performance Charts</h2>
                <img src="fps_comparison.png" alt="FPS Comparison">
                <img src="inference_time.png" alt="Inference Time">
                <img src="resource_usage.png" alt="Resource Usage">
                
                <h2>Recommendations</h2>
                <ul>
                    {recommendations}
                </ul>
            </div>
        </body>
        </html>
        """

        # Calculate best performers
        best_fps_idx = df['fps_mean'].idxmax()
        best_latency_idx = df['inference_time_ms_mean'].idxmin()

        # Generate recommendations
        recommendations = []

        # FPS recommendation
        if df.loc[best_fps_idx, 'fps_mean'] > 30:
            recommendations.append(
                f"<li>For real-time performance, use <strong>{df.loc[best_fps_idx, 'config']}</strong> "
                f"configuration which achieves {df.loc[best_fps_idx, 'fps_mean']:.1f} FPS</li>"
            )

        # Latency recommendation
        if df.loc[best_latency_idx, 'inference_time_ms_mean'] < 20:
            recommendations.append(
                f"<li>For lowest latency, use <strong>{df.loc[best_latency_idx, 'config']}</strong> "
                f"configuration with {df.loc[best_latency_idx, 'inference_time_ms_mean']:.1f} ms inference time</li>"
            )

        # Resource recommendation
        low_resource_configs = df[df['memory_usage_mb_mean'] < 1000]
        if not low_resource_configs.empty:
            best_low_resource = low_resource_configs.loc[low_resource_configs['fps_mean'].idxmax()]
            recommendations.append(
                f"<li>For resource-constrained environments, use <strong>{best_low_resource['config']}</strong> "
                f"which uses only {best_low_resource['memory_usage_mb_mean']:.0f} MB memory</li>"
            )

        # Fill template
        html_content = html_template.format(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            best_fps=df.loc[best_fps_idx, 'fps_mean'],
            best_fps_config=df.loc[best_fps_idx, 'config'],
            best_latency=df.loc[best_latency_idx, 'inference_time_ms_mean'],
            best_latency_config=df.loc[best_latency_idx, 'config'],
            best_efficiency=0,  # TODO: Calculate efficiency metric
            best_efficiency_config="N/A",
            results_table=df.to_html(index=False, float_format=lambda x: f'{x:.2f}'),
            recommendations='\n'.join(recommendations)
        )

        # Save HTML
        html_path = output_dir / 'report.html'
        with open(html_path, 'w') as f:
            f.write(html_content)

        logger.info(f"HTML report saved to: {html_path}")


def main():
    """Main entry point for benchmark"""
    import argparse

    parser = argparse.ArgumentParser(description='LIMA Traffic Counter Performance Benchmark')
    parser.add_argument('--video', type=str, help='Path to test video')
    parser.add_argument('--duration', type=int, default=60, help='Test duration in seconds')
    parser.add_argument('--models', nargs='+', help='Models to test')
    parser.add_argument('--devices', nargs='+', help='Devices to test')
    parser.add_argument('--output', type=str, default='benchmark_results', help='Output directory')

    args = parser.parse_args()

    # Create benchmark configuration
    config = BenchmarkConfig()

    if args.video:
        config.video_path = args.video
    config.duration_seconds = args.duration

    if args.models:
        config.models = args.models
    if args.devices:
        config.devices = args.devices

    config.output_dir = Path(args.output)

    # Run benchmark
    benchmark = PerformanceBenchmark(config)
    asyncio.run(benchmark.run())


if __name__ == "__main__":
    main()