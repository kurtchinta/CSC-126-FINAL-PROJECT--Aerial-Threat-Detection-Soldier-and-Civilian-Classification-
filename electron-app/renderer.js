const { ipcRenderer } = require('electron');

let currentTab = 'video';
let isDetectionRunning = false;

// Tab switching
function switchTab(tab, event) {
    currentTab = tab;
    
    // Update tab buttons
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    if (event && event.target) {
        event.target.classList.add('active');
    }
    
    // Update tab content
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.getElementById(`${tab}-tab`).classList.add('active');
    
    // Show/hide RTSP URL input
    if (tab === 'stream') {
        document.getElementById('streamSource').addEventListener('change', (e) => {
            document.getElementById('rtspUrlGroup').style.display = 
                e.target.value === 'rtsp' ? 'block' : 'none';
        });
    }
}

// File selection
async function selectVideoFile() {
    const filePath = await ipcRenderer.invoke('select-video-file');
    if (filePath) {
        document.getElementById('videoPath').value = filePath;
        addLog('Video file selected: ' + filePath, 'info');
    }
}

async function selectModelFile() {
    const filePath = await ipcRenderer.invoke('select-model-file');
    if (filePath) {
        document.getElementById('modelPath').value = filePath;
        addLog('Model file selected: ' + filePath, 'info');
    }
}

// Detection control
async function startDetection() {
    if (isDetectionRunning) {
        addLog('Detection already running', 'error');
        return;
    }
    
    const config = {
        confidence: parseFloat(document.getElementById('confidence').value),
        iou: parseFloat(document.getElementById('iou').value),
        modelPath: document.getElementById('modelPath').value
    };
    
    try {
        if (currentTab === 'video') {
            // Video detection
            const videoPath = document.getElementById('videoPath').value;
            if (!videoPath) {
                addLog('Please select a video file', 'error');
                return;
            }
            
            config.videoPath = videoPath;
            
            addLog('Starting video detection...', 'info');
            const result = await ipcRenderer.invoke('start-detection', config);
            
            if (result.success) {
                setDetectionRunning(true);
                addLog('Detection started successfully', 'success');
            } else {
                addLog('Failed to start detection: ' + result.message, 'error');
            }
        } else {
            // Stream detection
            const source = document.getElementById('streamSource').value;
            config.source = source === 'rtsp' ? 
                document.getElementById('rtspUrl').value : source;
            
            if (!config.source) {
                addLog('Please specify stream source', 'error');
                return;
            }
            
            addLog('Starting stream detection...', 'info');
            const result = await ipcRenderer.invoke('start-stream', config);
            
            if (result.success) {
                setDetectionRunning(true);
                addLog('Stream detection started successfully', 'success');
            } else {
                addLog('Failed to start stream detection: ' + result.message, 'error');
            }
        }
    } catch (error) {
        addLog('Error: ' + error.message, 'error');
    }
}

async function stopDetection() {
    try {
        addLog('Stopping detection...', 'info');
        const result = await ipcRenderer.invoke('stop-detection');
        
        if (result.success) {
            setDetectionRunning(false);
            addLog('Detection stopped', 'info');
        }
    } catch (error) {
        addLog('Error stopping detection: ' + error.message, 'error');
    }
}

// UI updates
function setDetectionRunning(running) {
    isDetectionRunning = running;
    
    document.getElementById('startBtn').disabled = running;
    document.getElementById('stopBtn').disabled = !running;
    
    const indicator = document.getElementById('statusIndicator');
    const statusText = document.getElementById('statusText');
    
    if (running) {
        indicator.className = 'status-indicator running';
        statusText.textContent = 'Detection Running';
    } else {
        indicator.className = 'status-indicator idle';
        statusText.textContent = 'Idle';
    }
}

function addLog(message, type = 'info') {
    const outputArea = document.getElementById('outputArea');
    const timestamp = new Date().toLocaleTimeString();
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry ${type}`;
    logEntry.textContent = `[${timestamp}] ${message}`;
    outputArea.appendChild(logEntry);
    outputArea.scrollTop = outputArea.scrollHeight;
}

function clearOutput() {
    document.getElementById('outputArea').innerHTML = 
        '<div class="log-entry info">Log cleared.</div>';
}

function updateStats(civilians, soldiers) {
    document.getElementById('civilianCount').textContent = civilians;
    document.getElementById('soldierCount').textContent = soldiers;
    document.getElementById('personCount').textContent = civilians + soldiers;
}

// IPC event listeners
ipcRenderer.on('detection-output', (event, data) => {
    addLog(data.trim(), 'info');
    
    // Parse detection counts from actual model output
    // Format: "Civilians: 5 | Soldiers: 3"
    const civilianMatch = data.match(/Civilians?:\s*(\d+)/i);
    const soldierMatch = data.match(/Soldiers?:\s*(\d+)/i);
    
    if (civilianMatch || soldierMatch) {
        const civilians = civilianMatch ? parseInt(civilianMatch[1]) : 0;
        const soldiers = soldierMatch ? parseInt(soldierMatch[1]) : 0;
        updateStats(civilians, soldiers);
    }
});

ipcRenderer.on('detection-error', (event, data) => {
    addLog(data.trim(), 'error');
});

ipcRenderer.on('detection-complete', (event, code) => {
    setDetectionRunning(false);
    if (code === 0) {
        addLog('Detection completed successfully', 'success');
    } else {
        addLog(`Detection ended with code ${code}`, 'error');
    }
});

// Initialize
window.addEventListener('DOMContentLoaded', async () => {
    addLog('Aerial Surveillance System initialized', 'success');
    
    // Check if model exists
    const modelExists = await ipcRenderer.invoke('check-model');
    if (!modelExists) {
        addLog('Warning: Default model not found. Please select a trained model.', 'error');
    } else {
        addLog('Default model found', 'info');
    }
    
    // Get project info
    const projectInfo = await ipcRenderer.invoke('get-project-info');
    addLog(`Project root: ${projectInfo.projectRoot}`, 'info');
});
