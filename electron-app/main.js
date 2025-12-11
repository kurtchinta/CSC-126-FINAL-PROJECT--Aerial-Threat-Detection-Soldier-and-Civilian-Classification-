const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

let mainWindow;
let detectionProcess = null;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    },
    icon: path.join(__dirname, 'assets', 'icon.png'),
    title: 'Aerial Surveillance Classification System'
  });

  mainWindow.loadFile('index.html');
  
  // Open DevTools in development
  // mainWindow.webContents.openDevTools();

  mainWindow.on('closed', () => {
    mainWindow = null;
    stopDetection();
  });
}

app.on('ready', createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});

// IPC Handlers

// Select video file
ipcMain.handle('select-video-file', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [
      { name: 'Videos', extensions: ['mp4', 'avi', 'mov', 'mkv'] }
    ]
  });
  
  if (!result.canceled && result.filePaths.length > 0) {
    return result.filePaths[0];
  }
  return null;
});

// Select model file
ipcMain.handle('select-model-file', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [
      { name: 'YOLO Models', extensions: ['pt'] }
    ]
  });
  
  if (!result.canceled && result.filePaths.length > 0) {
    return result.filePaths[0];
  }
  return null;
});

// Start detection
ipcMain.handle('start-detection', async (event, config) => {
  return new Promise((resolve, reject) => {
    try {
      // Get project root (parent directory of electron-app)
      const projectRoot = path.join(__dirname, '..');
      const pythonScript = path.join(projectRoot, 'src', 'detection', 'detect_video.py');
      
      // Build command
      const args = [
        pythonScript,
        '--source', config.videoPath,
        '--weights', config.modelPath || path.join(projectRoot, 'backend', 'civilian_soldier_working', 'yolo11n.pt'),
        '--conf', config.confidence.toString(),
        '--iou', config.iou.toString()
      ];
      
      if (config.outputPath) {
        args.push('--output', config.outputPath);
      }
      
      // Spawn Python process
      detectionProcess = spawn('python', args, {
        cwd: projectRoot
      });
      
      // Send output to renderer
      detectionProcess.stdout.on('data', (data) => {
        mainWindow.webContents.send('detection-output', data.toString());
      });
      
      detectionProcess.stderr.on('data', (data) => {
        mainWindow.webContents.send('detection-error', data.toString());
      });
      
      detectionProcess.on('close', (code) => {
        mainWindow.webContents.send('detection-complete', code);
        detectionProcess = null;
      });
      
      resolve({ success: true, message: 'Detection started' });
      
    } catch (error) {
      reject({ success: false, message: error.message });
    }
  });
});

// Start stream detection
ipcMain.handle('start-stream', async (event, config) => {
  return new Promise((resolve, reject) => {
    try {
      const projectRoot = path.join(__dirname, '..');
      const pythonScript = path.join(projectRoot, 'src', 'detection', 'detect_stream.py');
      
      const args = [
        pythonScript,
        '--source', config.source || '0',
        '--weights', config.modelPath || path.join(projectRoot, 'backend', 'civilian_soldier_working', 'yolo11n.pt'),
        '--conf', config.confidence.toString(),
        '--iou', config.iou.toString()
      ];
      
      detectionProcess = spawn('python', args, {
        cwd: projectRoot
      });
      
      detectionProcess.stdout.on('data', (data) => {
        mainWindow.webContents.send('detection-output', data.toString());
      });
      
      detectionProcess.stderr.on('data', (data) => {
        mainWindow.webContents.send('detection-error', data.toString());
      });
      
      detectionProcess.on('close', (code) => {
        mainWindow.webContents.send('detection-complete', code);
        detectionProcess = null;
      });
      
      resolve({ success: true, message: 'Stream detection started' });
      
    } catch (error) {
      reject({ success: false, message: error.message });
    }
  });
});

// Stop detection
ipcMain.handle('stop-detection', async () => {
  stopDetection();
  return { success: true, message: 'Detection stopped' };
});

function stopDetection() {
  if (detectionProcess) {
    detectionProcess.kill();
    detectionProcess = null;
  }
}

// Check if model exists
ipcMain.handle('check-model', async () => {
  const projectRoot = path.join(__dirname, '..');
  const modelPath = path.join(projectRoot, 'backend', 'civilian_soldier_working', 'yolo11n.pt');
  return fs.existsSync(modelPath);
});

// Get project info
ipcMain.handle('get-project-info', async () => {
  const projectRoot = path.join(__dirname, '..');
  return {
    projectRoot,
    modelsDir: path.join(projectRoot, 'models'),
    dataDir: path.join(projectRoot, 'data'),
    outputDir: path.join(projectRoot, 'output')
  };
});
