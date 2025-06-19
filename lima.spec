# lima.spec
# PyInstaller build specification for LIMA Traffic Counter

import sys
from PyInstaller.utils.hooks import collect_all, collect_submodules

# Collect all dependencies
datas = []
binaries = []
hiddenimports = []

# Collect OpenVINO
openvino_datas, openvino_binaries, openvino_hiddenimports = collect_all('openvino')
datas += openvino_datas
binaries += openvino_binaries
hiddenimports += openvino_hiddenimports

# Collect PySide6
pyside6_datas, pyside6_binaries, pyside6_hiddenimports = collect_all('PySide6')
datas += pyside6_datas
binaries += pyside6_binaries
hiddenimports += pyside6_hiddenimports

# Collect other dependencies
hiddenimports += collect_submodules('cv2')
hiddenimports += collect_submodules('numpy')
hiddenimports += collect_submodules('pandas')
hiddenimports += collect_submodules('scipy')
hiddenimports += collect_submodules('pyqtgraph')
hiddenimports += collect_submodules('aiohttp')
hiddenimports += collect_submodules('aiosqlite')
hiddenimports += collect_submodules('structlog')

# Add custom data files
datas += [
    ('resources', 'resources'),
    ('models', 'models'),
    ('.env.example', '.'),
    ('README.md', '.'),
]

a = Analysis(
    ['src/main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + [
        'qasync',
        'xlsxwriter',
        'openpyxl',
        'reportlab',
        'matplotlib',
        'seaborn',
        'plotly',
        'PIL',
        'psutil',
        'click',
        'pydantic',
        'dotenv',
        'backoff',
        'pythonjsonlogger',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'test',
        'tests',
        'testing',
        'unittest',
        'pytest',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='LIMA-Traffic-Counter',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Set to False for GUI app
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='resources/icons/app_icon.ico' if sys.platform == 'win32' else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='LIMA-Traffic-Counter',
)

# For macOS, create .app bundle
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='LIMA Traffic Counter.app',
        icon='resources/icons/app_icon.icns',
        bundle_identifier='com.lintasmediatama.lima',
        info_plist={
            'NSHighResolutionCapable': 'True',
            'LSBackgroundOnly': 'False',
        },
    )