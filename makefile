
---

# Makefile
.PHONY: help install install-dev clean test format lint build docker-build docker-run

help:
@echo "Available commands:"
@echo "  install       Install the package"
@echo "  install-dev   Install with development dependencies"
@echo "  clean         Clean build artifacts"
@echo "  test          Run tests"
@echo "  format        Format code with black and isort"
@echo "  lint          Run linting checks"
@echo "  build         Build distribution packages"
@echo "  docker-build  Build Docker image"
@echo "  docker-run    Run Docker container"

install:
pip install -e .

install-dev:
pip install -e ".[dev]"
pre-commit install

clean:
rm -rf build/
rm -rf dist/
rm -rf *.egg-info
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

test:
pytest tests/ -v --cov=src --cov-report=html

format:
black src/ tests/
isort src/ tests/

lint:
black --check src/ tests/
isort --check-only src/ tests/
mypy src/
flake8 src/ tests/

build: clean
python -m build

docker-build:
docker build -t lima-traffic-counter:latest .

docker-run:
docker run -it --rm \
--gpus all \
-v $(PWD)/data:/app/data \
-v $(PWD)/models:/app/models \
-p 8000:8000 \
lima-traffic-counter:latest

---

# .gitignore
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
.pybuilder/
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/

# Project specific
data/
models/*.onnx
models/*.xml
models/*.bin
models/*.engine
*.db
logs/
temp/
benchmark_results/
.lima/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

---
