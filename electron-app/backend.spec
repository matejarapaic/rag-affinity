# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for the Affinity AI backend
# Run from the project root: pyinstaller electron-app/backend.spec
#
# Output: electron-app/resources/backend/   (single-folder dist)

import os
import sys

# Absolute path to the backend source directory
BACKEND_DIR = os.path.abspath(os.path.join(SPECPATH, '..', 'backend'))
ENTRY_POINT = os.path.join(BACKEND_DIR, 'main.py')

block_cipher = None

a = Analysis(
    [ENTRY_POINT],
    pathex=[BACKEND_DIR],
    binaries=[],
    datas=[
        # Include any data files the backend needs at runtime
        (os.path.join(BACKEND_DIR, 'config.py'), '.'),
        # Bundle the .env so the frozen binary can find the API keys
        (os.path.join(BACKEND_DIR, '.env'), '.'),
        # Google Drive service account credentials
        (os.path.join(BACKEND_DIR, 'service_account.json'), '.'),
        # Frontend static files — served by FastAPI at http://localhost:8000
        (os.path.abspath(os.path.join(SPECPATH, '..', 'frontend')), 'frontend'),
    ],
    hiddenimports=[
        # FastAPI / uvicorn stack
        'uvicorn',
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.loops.asyncio',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.http.h11_impl',
        'uvicorn.protocols.http.httptools_impl',
        'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
        'fastapi',
        'fastapi.middleware.cors',
        'starlette',
        'starlette.middleware',
        'starlette.middleware.cors',
        'pydantic',
        'pydantic.v1',
        'anyio',
        'anyio._backends._asyncio',
        # AI / ML
        'anthropic',
        'fastembed',
        'qdrant_client',
        'qdrant_client.http',
        'qdrant_client.http.models',
        # Graph / NLP
        'networkx',
        'spacy',
        'spacy.lang.en',
        # Google APIs
        'google.oauth2',
        'google.oauth2.credentials',
        'google.oauth2.service_account',
        'googleapiclient',
        'googleapiclient.discovery',
        'googleapiclient.http',
        'google.auth',
        'google.auth.transport.requests',
        # Rate limiting
        'slowapi',
        'slowapi.util',
        'slowapi.errors',
        'limits',
        'limits.storage',
        'limits.strategies',
        # Auth / Clerk
        'jwt',
        'jwt.algorithms',
        'cryptography',
        'cryptography.hazmat.primitives.asymmetric.rsa',
        # File upload
        'multipart',
        'python_multipart',
        # Misc
        'email.mime.text',
        'email.mime.multipart',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='backend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='backend',
)
