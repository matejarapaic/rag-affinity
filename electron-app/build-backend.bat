@echo off
REM build-backend.bat — Bundle the Python backend with PyInstaller (Windows)
REM
REM Usage (run from project root or electron-app\):
REM   build-backend.bat
REM
REM After this script completes, electron-app\resources\backend\ will contain
REM the compiled .exe. Then run: npm run dist  (from electron-app\)

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
REM Remove trailing backslash
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "PROJECT_ROOT=%SCRIPT_DIR%\.."

echo =^> Project root: %PROJECT_ROOT%
echo =^> Spec file:    %SCRIPT_DIR%\backend.spec

REM Prefer project venv Python if available
set "VENV_PYTHON=%PROJECT_ROOT%\rag-chatbot\backend\.venv\Scripts\python.exe"
if exist "%VENV_PYTHON%" (
    set "PYTHON=%VENV_PYTHON%"
    echo =^> Using venv Python: %PYTHON%
) else (
    set "PYTHON=python"
    echo =^> Using system Python
)

REM Ensure PyInstaller is available
%PYTHON% -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
    echo =^> Installing PyInstaller...
    %PYTHON% -m pip install pyinstaller
)

REM Clean previous build artifacts
if exist "%SCRIPT_DIR%\resources\backend" rmdir /s /q "%SCRIPT_DIR%\resources\backend"
if exist "%PROJECT_ROOT%\build\backend" rmdir /s /q "%PROJECT_ROOT%\build\backend"

echo =^> Running PyInstaller...
%PYTHON% -m PyInstaller ^
  --distpath "%SCRIPT_DIR%\resources" ^
  --workpath "%PROJECT_ROOT%\build" ^
  --noconfirm ^
  "%SCRIPT_DIR%\backend.spec"

if errorlevel 1 (
    echo ERROR: PyInstaller failed.
    exit /b 1
)

echo.
echo =^> Build complete!
echo     Binary output: %SCRIPT_DIR%\resources\backend\
echo.
echo     Next step: npm run dist
