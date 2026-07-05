# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = [('static', 'static')]
binaries = []
hiddenimports = [
    'graph', 'main',
    # uvicorn
    'uvicorn.logging', 'uvicorn.loops', 'uvicorn.loops.auto',
    'uvicorn.loops.asyncio', 'uvicorn.protocols', 'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto', 'uvicorn.protocols.http.h11_impl',
    'uvicorn.protocols.websockets', 'uvicorn.protocols.websockets.auto',
    'uvicorn.protocols.websockets.websockets_impl',
    'uvicorn.lifespan', 'uvicorn.lifespan.on',
    # starlette
    'starlette.routing', 'starlette.middleware', 'starlette.staticfiles',
    'starlette.responses', 'starlette.background', 'starlette.concurrency',
    # pydantic
    'pydantic', 'pydantic.v1',
    # misc
    'h11', 'anyio', 'anyio._backends._asyncio', 'sniffio',
    'email.mime.text', 'email.mime.multipart',
    'multipart', 'python_multipart',
    'dotenv',
]

# langchain / langgraph / openai 전체 수집
for pkg in ['langchain_core', 'langchain_openai', 'langgraph', 'openai', 'httpx', 'httpcore']:
    try:
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hiddenimports += h
    except Exception:
        pass

hiddenimports += collect_submodules('langchain_core')
hiddenimports += collect_submodules('langchain_openai')
hiddenimports += collect_submodules('langgraph')

a = Analysis(
    ['launcher.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'numpy', 'pandas', 'scipy', 'PIL',
              'cv2', 'torch', 'tensorflow', 'pytest'],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SpectrumDashboard',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='SpectrumDashboard',
)
