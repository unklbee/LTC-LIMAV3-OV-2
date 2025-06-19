
# README.md
# LIMA Traffic Counter

Advanced traffic counting system with AI-powered vehicle detection and tracking.

## Features

- 🚀 **High Performance**: Optimized pipeline with support for TensorRT, OpenVINO, and ONNX Runtime
- 🎯 **Accurate Detection**: State-of-the-art object detection with YOLOv7/v8 models
- 🔄 **Advanced Tracking**: ByteTrack algorithm for robust multi-object tracking
- 📊 **Real-time Analytics**: Live dashboard with statistics and visualizations
- 💾 **Data Management**: SQLite database with automatic backups and API integration
- 🎨 **Modern UI**: Beautiful dark theme with smooth animations
- 🌍 **Multi-language**: Support for English, Indonesian, and Chinese
- 📱 **Cross-platform**: Works on Windows, Linux, and macOS

## Installation

### Requirements

- Python 3.10 or higher
- OpenVINO 2024.0+ (for Intel hardware acceleration)
- CUDA 11.8+ (optional, for NVIDIA GPU support)
- TensorRT 8.6+ (optional, for maximum GPU performance)

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/lintasmediatama/lima-traffic-counter.git
cd lima-traffic-counter
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

4. Download models:
```bash
python scripts/download_models.py
```

5. Run the application:
```bash
lima-counter
```

## Usage

### GUI Application

```bash
lima-counter
```

### Command Line Interface

```bash
lima-counter-cli --source video.mp4 --model yolov7-tiny
```

### API Server

```bash
lima-counter-server --port 8000
```

### Performance Benchmark

```bash
lima-counter-benchmark --video test.mp4 --duration 60
```

## Configuration

Create a `.env` file in the project root:

```env
LIMA_MODEL_DIR=./models
LIMA_DATA_DIR=./data
LIMA_DB_PATH=./data/counts.db
LIMA_USE_GPU=true
LIMA_API_URL=https://api.example.com/traffic
LIMA_API_KEY=your-api-key
```

## Docker

Build and run with Docker:

```bash
docker build -t lima-traffic-counter .
docker run -it --rm --gpus all -v $(pwd)/data:/app/data lima-traffic-counter
```

## Development

### Setup Development Environment

```bash
pip install -e ".[dev]"
pre-commit install
```

### Run Tests

```bash
pytest tests/ -v --cov=src
```

### Format Code

```bash
make format
```

### Build Documentation

```bash
cd docs
make html
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv7/v8 for object detection
- ByteTrack for object tracking
- OpenVINO for Intel optimization
- TensorRT for NVIDIA optimization

## Support

For support and questions:
- 📧 Email: support@lintasmediatama.com
- 🐛 Issues: [GitHub Issues](https://github.com/lintasmediatama/lima-traffic-counter/issues)
- 📖 Docs: [Documentation](https://lima-traffic-counter.readthedocs.io)