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
  
  // Open DevTools for debugging
  mainWindow.webContents.openDevTools();

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

// Start detection (video with Electron display)
ipcMain.handle('start-detection', async (event, config) => {
  return new Promise((resolve, reject) => {
    try {
      // Get project root (parent directory of electron-app)
      const projectRoot = path.join(__dirname, '..');
      const pythonScript = path.join(projectRoot, 'src', 'detection', 'detect_video.py');
      
      // Build command with unified script arguments
      const args = [
        pythonScript,
        '--source', config.videoPath,
        '--mode', 'video',
        '--conf', config.confidence.toString(),
        '--iou', config.iou.toString(),
        '--electron'  // Enable Flask streaming for Electron
      ];
      
      // Add weights if custom model specified
      if (config.modelPath) {
        args.push('--weights', path.join(projectRoot, config.modelPath));
      }
      
      // Spawn Python process
      console.log('Spawning Python process...');
      console.log('Python script:', pythonScript);
      console.log('Args:', args);
      
      detectionProcess = spawn('python', args, {
        cwd: projectRoot,
        shell: true
      });
      
      console.log('Process spawned with PID:', detectionProcess.pid);
      
      let startupComplete = false;
      
      // Send output to renderer
      detectionProcess.stdout.on('data', (data) => {
        const output = data.toString();
        console.log('Python stdout:', output);
        mainWindow.webContents.send('detection-output', output);
        
        // Check if Flask server started
        if (!startupComplete && output.includes('Flask server started')) {
          startupComplete = true;
          console.log('Flask server detected as started');
          setTimeout(() => {
            mainWindow.webContents.send('flask-ready');
          }, 1500);
        }
      });
      
      detectionProcess.stderr.on('data', (data) => {
        const error = data.toString();
        console.error('Python stderr:', error);
        mainWindow.webContents.send('detection-error', error);
      });
      
      detectionProcess.on('error', (error) => {
        console.error('Failed to start Python process:', error);
        mainWindow.webContents.send('detection-error', `Failed to start: ${error.message}`);
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
      const pythonScript = path.join(projectRoot, 'src', 'detection', 'detect_video.py');
      
      const args = [
        pythonScript,
        '--source', config.source || '0',
        '--mode', 'stream',
        '--conf', config.confidence.toString(),
        '--iou', config.iou.toString()
      ];
      
      // Add weights if custom model specified
      if (config.modelPath) {
        args.push('--weights', config.modelPath);
      }
      
      detectionProcess = spawn('python', args, {
        cwd: projectRoot,
        shell: true
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

// Start stream detection with Electron display
ipcMain.handle('start-stream-electron', async (event, config) => {
  return new Promise((resolve, reject) => {
    try {
      const projectRoot = path.join(__dirname, '..');
      const pythonScript = path.join(projectRoot, 'src', 'detection', 'detect_video.py');
      
      const args = [
        pythonScript,
        '--source', config.source || '0',
        '--mode', 'stream',
        '--conf', config.confidence.toString(),
        '--iou', config.iou.toString(),
        '--electron'  // Enable Flask streaming for Electron
      ];
      
      // Add weights if custom model specified
      if (config.modelPath) {
        args.push('--weights', config.modelPath);
      }
      
      detectionProcess = spawn('python', args, {
        cwd: projectRoot,
        shell: true
      });
      
      let streamStarted = false;
      
      detectionProcess.stdout.on('data', (data) => {
        const output = data.toString();
        mainWindow.webContents.send('detection-output', output);
        
        // Check if Flask server started
        if (!streamStarted && output.includes('Flask server started')) {
          streamStarted = true;
          setTimeout(() => {
            mainWindow.webContents.send('flask-ready');
          }, 1500);
        }
      });
      
      detectionProcess.stderr.on('data', (data) => {
        mainWindow.webContents.send('detection-error', data.toString());
      });
      
      detectionProcess.on('close', (code) => {
        mainWindow.webContents.send('detection-complete', code);
        detectionProcess = null;
      });
      
      resolve({ success: true, message: 'Stream detection started with dual display' });
      
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
