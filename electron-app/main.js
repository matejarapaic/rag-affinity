const { app, BrowserWindow, shell } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const http = require('http');
const fs = require('fs');

let mainWindow = null;
let splashWindow = null;
let backendProcess = null;
let qdrantProcess = null;

// ── Path resolution ──────────────────────────────────────────────────────────

function getBackendPath() {
  if (app.isPackaged) {
    const exe = process.platform === 'win32' ? 'backend.exe' : 'backend';
    return path.join(process.resourcesPath, 'backend', exe);
  }
  return null; // dev: use venv Python
}

function getQdrantPath() {
  if (app.isPackaged) {
    const exe = process.platform === 'win32' ? 'qdrant.exe' : 'qdrant';
    return path.join(process.resourcesPath, 'qdrant', exe);
  }
  // Dev: look for the binary in resources/qdrant/ alongside electron-app/
  const exe = process.platform === 'win32' ? 'qdrant.exe' : 'qdrant';
  return path.join(__dirname, 'resources', 'qdrant', exe);
}

// Frontend is now served by the FastAPI backend at http://localhost:8000
// (required for Clerk auth — Clerk does not work on file:// origins)

// ── Qdrant spawning ───────────────────────────────────────────────────────────

function startQdrant() {
  const qdrantBin = getQdrantPath();

  if (!fs.existsSync(qdrantBin)) {
    console.warn('[qdrant] Binary not found at:', qdrantBin, '— skipping (Docker/external Qdrant assumed)');
    return;
  }

  // Persist Qdrant data in the user's app data folder so it survives restarts
  const storagePath = path.join(app.getPath('userData'), 'qdrant_storage');
  fs.mkdirSync(storagePath, { recursive: true });

  console.log('[qdrant] Starting binary:', qdrantBin);
  console.log('[qdrant] Storage path:', storagePath);

  qdrantProcess = spawn(qdrantBin, [], {
    cwd: path.dirname(qdrantBin),
    env: {
      ...process.env,
      QDRANT__STORAGE__STORAGE_PATH: storagePath,
      QDRANT__SERVICE__HTTP_PORT: '6333',
      QDRANT__SERVICE__GRPC_PORT: '6334',
      QDRANT__LOG_LEVEL: 'WARN',
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  qdrantProcess.stdout.on('data', (d) => console.log('[qdrant stdout]', d.toString().trim()));
  qdrantProcess.stderr.on('data', (d) => console.log('[qdrant stderr]', d.toString().trim()));
  qdrantProcess.on('error', (err) => console.error('[qdrant] Failed to start:', err));
  qdrantProcess.on('exit', (code, signal) => {
    console.log(`[qdrant] Exited code=${code} signal=${signal}`);
    qdrantProcess = null;
  });
}

function killQdrant() {
  if (qdrantProcess) {
    console.log('[qdrant] Killing Qdrant process...');
    qdrantProcess.kill('SIGTERM');
    qdrantProcess = null;
  }
}

function waitForQdrant(timeout = 20000, interval = 500) {
  return new Promise((resolve, reject) => {
    const start = Date.now();

    // If the binary wasn't found we skip waiting
    const qdrantBin = getQdrantPath();
    if (!fs.existsSync(qdrantBin)) {
      resolve();
      return;
    }

    function check() {
      const req = http.get('http://localhost:6333/healthz', (res) => {
        if (res.statusCode === 200) {
          console.log('[qdrant] Health check passed.');
          resolve();
        } else {
          retry();
        }
        res.resume();
      });

      req.on('error', () => retry());
      req.setTimeout(400, () => { req.destroy(); retry(); });
    }

    function retry() {
      if (Date.now() - start >= timeout) {
        reject(new Error('Qdrant health check timed out after 20s'));
        return;
      }
      setTimeout(check, interval);
    }

    check();
  });
}

// ── Backend spawning ─────────────────────────────────────────────────────────

function startBackend() {
  const binaryPath = getBackendPath();

  if (binaryPath) {
    console.log('[backend] Starting packaged binary:', binaryPath);
    backendProcess = spawn(binaryPath, [], {
      cwd: path.dirname(binaryPath),
      stdio: ['ignore', 'pipe', 'pipe'],
    });
  } else {
    const venvPython = path.join(
      __dirname, '..', 'rag-chatbot', 'backend', '.venv', 'bin', 'python'
    );
    const mainPy = path.join(__dirname, '..', 'backend', 'main.py');
    const backendDir = path.join(__dirname, '..', 'backend');

    console.log('[backend] Starting dev server with:', venvPython, mainPy);
    backendProcess = spawn(venvPython, [mainPy], {
      cwd: backendDir,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
  }

  backendProcess.stdout.on('data', (data) => {
    console.log('[backend stdout]', data.toString().trim());
  });

  backendProcess.stderr.on('data', (data) => {
    console.error('[backend stderr]', data.toString().trim());
  });

  backendProcess.on('error', (err) => {
    console.error('[backend] Failed to start:', err);
  });

  backendProcess.on('exit', (code, signal) => {
    console.log(`[backend] Exited with code=${code} signal=${signal}`);
    backendProcess = null;
  });
}

function killBackend() {
  if (backendProcess) {
    console.log('[backend] Killing backend process...');
    backendProcess.kill('SIGTERM');
    backendProcess = null;
  }
}

// ── Health polling ───────────────────────────────────────────────────────────

function waitForBackend(timeout = 120000, interval = 500) {
  return new Promise((resolve, reject) => {
    const start = Date.now();

    function check() {
      const req = http.get('http://localhost:8000/health', (res) => {
        if (res.statusCode === 200) {
          console.log('[backend] Health check passed.');
          resolve();
        } else {
          retry();
        }
        res.resume();
      });

      req.on('error', () => retry());
      req.setTimeout(400, () => { req.destroy(); retry(); });
    }

    function retry() {
      if (Date.now() - start >= timeout) {
        reject(new Error('Backend health check timed out after 30s'));
        return;
      }
      setTimeout(check, interval);
    }

    check();
  });
}

// ── Windows ───────────────────────────────────────────────────────────────────

function createSplash() {
  splashWindow = new BrowserWindow({
    width: 480,
    height: 320,
    frame: false,
    transparent: false,
    resizable: false,
    alwaysOnTop: true,
    center: true,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  splashWindow.loadFile(path.join(__dirname, 'splash.html'));
  splashWindow.on('closed', () => { splashWindow = null; });
}

function setSplashStatus(msg) {
  if (splashWindow && !splashWindow.isDestroyed()) {
    splashWindow.webContents.executeJavaScript(
      `document.getElementById('status') && (document.getElementById('status').textContent = ${JSON.stringify(msg)})`
    ).catch(() => {});
  }
}

function createMainWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 900,
    minHeight: 600,
    show: false,
    title: 'Affinity RAG',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      webSecurity: false,
    },
  });

  mainWindow.loadURL('http://localhost:8000');

  mainWindow.once('ready-to-show', () => {
    if (splashWindow && !splashWindow.isDestroyed()) {
      splashWindow.close();
    }
    mainWindow.show();
    mainWindow.focus();
  });

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });

  mainWindow.on('closed', () => { mainWindow = null; });
}

// ── App lifecycle ─────────────────────────────────────────────────────────────

app.whenReady().then(async () => {
  createSplash();

  try {
    // 1. Start Qdrant first
    setSplashStatus('Starting database…');
    startQdrant();
    await waitForQdrant(20000, 500);

    // 2. Then start the Python backend (which connects to Qdrant on startup)
    setSplashStatus('Starting AI engine… (first launch may take 1–2 min)');
    startBackend();
    await waitForBackend(120000, 500);

    // 3. Open the main window
    setSplashStatus('Ready!');
    createMainWindow();
  } catch (err) {
    console.error('[app] Startup failed:', err.message);
    setSplashStatus(`Error: ${err.message}`);
    setTimeout(() => app.quit(), 5000);
  }
});

app.on('window-all-closed', () => {
  killBackend();
  killQdrant();
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createMainWindow();
  }
});

app.on('before-quit', () => {
  killBackend();
  killQdrant();
});

app.on('will-quit', () => {
  killBackend();
  killQdrant();
});
