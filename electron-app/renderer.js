const { ipcRenderer } = require('electron');

let currentTab = 'video';
let isDetectionRunning = false;
let statsInterval = null;

// Cumulative detection counts - only increment, never decrease
let totalCivilians = 0;
let totalSoldiers = 0;

// Tab switching
window.switchTab = function(tab, event) {
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
window.selectVideoFile = async function() {
    try {
        const filePath = await ipcRenderer.invoke('select-video-file');
        if (filePath) {
            document.getElementById('videoPath').value = filePath;
            addLog('Video file selected: ' + filePath, 'info');
            console.log('Video file selected:', filePath);
        }
    } catch (error) {
        addLog('Error selecting video: ' + error.message, 'error');
        console.error('Error selecting video:', error);
    }
}

window.selectModelFile = async function() {
    const filePath = await ipcRenderer.invoke('select-model-file');
    if (filePath) {
        document.getElementById('modelPath').value = filePath;
        addLog('Model file selected: ' + filePath, 'info');
    }
}

// Track if we're continuing (to preserve counts)
let isContinuing = false;

// Detection control
window.startDetection = async function() {
    if (isDetectionRunning) {
        addLog('Detection already running', 'error');
        return;
    }
    
    // Only reset counts if not continuing
    if (!isContinuing) {
        resetCumulativeCounts();
        document.getElementById('civilianCount').textContent = '0';
        document.getElementById('soldierCount').textContent = '0';
        document.getElementById('personCount').textContent = '0';
    }
    isContinuing = false;  // Reset flag
    
    const config = {
        confidence: parseFloat(document.getElementById('confidence').value),
        iou: parseFloat(document.getElementById('iou').value),
        modelPath: document.getElementById('modelPath').value
    };
    
    console.log('Starting detection with config:', config);
    
    try {
        if (currentTab === 'video') {
            // Video detection (existing functionality)
            const videoPath = document.getElementById('videoPath').value;
            console.log('Video path:', videoPath);
            
            if (!videoPath) {
                addLog('Please select a video file', 'error');
                return;
            }
            
            config.videoPath = videoPath;
            
            addLog('Starting video detection...', 'info');
            console.log('Invoking start-detection with config:', config);
            
            const result = await ipcRenderer.invoke('start-detection', config);
            console.log('Start detection result:', result);
            
            if (result.success) {
                setDetectionRunning(true);
                addLog('Detection started successfully', 'success');
                
                // Show video display section for video mode too
                document.getElementById('videoDisplaySection').style.display = 'flex';
                
                // Wait for Flask server to be ready before starting streams
                addLog('Waiting for Flask server to initialize...', 'info');
                
                // Stats polling will start when Flask is ready
            } else {
                addLog('Failed to start detection: ' + result.message, 'error');
            }
        } else {
            // Stream detection with dual display
            const source = document.getElementById('streamSource').value;
            config.source = source === 'rtsp' ? 
                document.getElementById('rtspUrl').value : source;
            
            if (!config.source) {
                addLog('Please specify stream source', 'error');
                return;
            }
            
            addLog('Starting stream detection...', 'info');
            const result = await ipcRenderer.invoke('start-stream-electron', config);
            
            if (result.success) {
                setDetectionRunning(true);
                addLog('Stream detection started successfully', 'success');
                
                // Show video display section
                document.getElementById('videoDisplaySection').style.display = 'flex';
                
                // Wait for Flask server to be ready
                addLog('Waiting for Flask server to initialize...', 'info');
                
                // Stats polling will start when Flask is ready
            } else {
                addLog('Failed to start stream detection: ' + result.message, 'error');
            }
        }
    } catch (error) {
        addLog('Error: ' + error.message, 'error');
    }
}

window.stopDetection = async function() {
    try {
        addLog('Stopping detection...', 'info');
        const result = await ipcRenderer.invoke('stop-detection');
        
        if (result.success) {
            setDetectionRunning(false);
            addLog('Detection stopped by user', 'info');
            
            // Stop streams and polling
            stopVideoStreams();
            stopStatsPolling();
            
            // Show stopped overlay
            showDetectionStoppedOverlay();
        }
    } catch (error) {
        addLog('Error stopping detection: ' + error.message, 'error');
    }
}

// Video streaming functions
function startVideoStreams() {
    const originalImg = document.getElementById('originalStreamImg');
    const detectedImg = document.getElementById('detectedStreamImg');
    const originalPlaceholder = document.getElementById('originalPlaceholder');
    const detectedPlaceholder = document.getElementById('detectedPlaceholder');
    
    // Update placeholder messages to "Processing..."
    originalPlaceholder.textContent = 'Processing video...';
    detectedPlaceholder.textContent = 'Running detection...';
    originalPlaceholder.style.color = '#aaa';
    detectedPlaceholder.style.color = '#aaa';
    originalPlaceholder.style.fontSize = '13px';
    detectedPlaceholder.style.fontSize = '13px';
    
    // Add error handlers to suppress console errors
    originalImg.onerror = () => {
        // Silently handle image load errors
    };
    detectedImg.onerror = () => {
        // Silently handle image load errors
    };
    
    // Set stream URLs with timestamp to prevent caching
    const timestamp = new Date().getTime();
    originalImg.src = `http://127.0.0.1:5000/video/original?t=${timestamp}`;
    detectedImg.src = `http://127.0.0.1:5000/video/detected?t=${timestamp}`;
    
    // Show images, hide placeholders
    originalImg.style.display = 'block';
    detectedImg.style.display = 'block';
    originalPlaceholder.style.display = 'none';
    detectedPlaceholder.style.display = 'none';
    
    addLog('Video streams connected', 'success');
}

function stopVideoStreams() {
    const originalImg = document.getElementById('originalStreamImg');
    const detectedImg = document.getElementById('detectedStreamImg');
    const originalPlaceholder = document.getElementById('originalPlaceholder');
    const detectedPlaceholder = document.getElementById('detectedPlaceholder');
    
    // Clear stream sources
    originalImg.src = '';
    detectedImg.src = '';
    
    // Hide images, show placeholders
    originalImg.style.display = 'none';
    detectedImg.style.display = 'none';
    originalPlaceholder.style.display = 'block';
    detectedPlaceholder.style.display = 'block';
}

function startStatsPolling() {
    let lastLogTime = 0;
    let connectionErrors = 0;
    const maxConnectionErrors = 3;  // Stop polling after 3 consecutive errors
    
    // Poll statistics every 200ms for real-time updates
    statsInterval = setInterval(async () => {
        // Don't poll if we've had too many connection errors
        if (connectionErrors >= maxConnectionErrors) {
            stopStatsPolling();
            return;
        }
        
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 1000);  // 1 second timeout
            
            const response = await fetch('http://127.0.0.1:5000/stats', {
                signal: controller.signal
            });
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                connectionErrors++;
                return;
            }
            
            // Reset error count on successful response
            connectionErrors = 0;
            
            const stats = await response.json();
            
            // Check status field (processing, streaming, complete, stopped)
            const isActive = stats.status === 'processing' || stats.status === 'streaming';
            
            if (isActive || stats.frame_count > 0) {
                // Update stat cards immediately (real-time) - this shows in Detection Statistics section
                updateStats(stats.civilians || 0, stats.soldiers || 0);
                
                // Display detailed statistics every 1 second
                const currentTime = Date.now();
                if (currentTime - lastLogTime >= 1000 && isActive) {
                    const fps = stats.fps || 0;
                    const progress = stats.progress || 0;
                    const frameCount = stats.frame_count || 0;
                    const totalFrames = stats.total_frames || 0;
                    const totalDetections = (stats.civilians || 0) + (stats.soldiers || 0);
                    
                    const statsMsg = `REAL-TIME DETECTION DATA
Civilians: ${stats.civilians || 0} | Soldiers: ${stats.soldiers || 0} | Total: ${totalDetections}
FPS: ${fps.toFixed(1)} | Frame: ${frameCount}${totalFrames ? '/' + totalFrames : ''} | Progress: ${progress.toFixed(1)}%
Time: ${new Date().toLocaleTimeString()}`;
                    
                    addLog(statsMsg, 'info');
                    lastLogTime = currentTime;
                }
            }
            
            // Stop polling if status indicates completion
            if (stats.status === 'complete' || stats.status === 'stopped') {
                stopStatsPolling();
            }
        } catch (error) {
            // Increment error count - after too many errors, polling will stop
            connectionErrors++;
            // Silently fail if server not responding - no logging to keep console clean
        }
    }, 200);
}

function stopStatsPolling() {
    if (statsInterval) {
        clearInterval(statsInterval);
        statsInterval = null;
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

window.clearOutput = function() {
    document.getElementById('outputArea').innerHTML = 
        '<div class="log-entry info">Log cleared.</div>';
}

function updateStats(civilians, soldiers) {
    // Only update if the new count is higher (cumulative tracking)
    if (civilians > totalCivilians) {
        totalCivilians = civilians;
    }
    if (soldiers > totalSoldiers) {
        totalSoldiers = soldiers;
    }
    
    // Display the cumulative totals
    document.getElementById('civilianCount').textContent = totalCivilians;
    document.getElementById('soldierCount').textContent = totalSoldiers;
    document.getElementById('personCount').textContent = totalCivilians + totalSoldiers;
}

// Reset cumulative counts (called when starting new detection)
function resetCumulativeCounts() {
    totalCivilians = 0;
    totalSoldiers = 0;
}

// IPC event listeners
ipcRenderer.on('flask-ready', () => {
    addLog('Flask server ready, connecting video streams...', 'success');
    if (document.getElementById('videoDisplaySection').style.display === 'flex') {
        startVideoStreams();
        startStatsPolling();
    }
});

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
    
    // Stop polling and video streams immediately
    stopStatsPolling();
    stopVideoStreams();
    
    // Show completion message in video frames
    const originalPlaceholder = document.getElementById('originalPlaceholder');
    const detectedPlaceholder = document.getElementById('detectedPlaceholder');
    
    originalPlaceholder.textContent = 'Video processing complete';
    detectedPlaceholder.textContent = 'Detection complete';
    originalPlaceholder.style.color = '#10b981';
    detectedPlaceholder.style.color = '#10b981';
    originalPlaceholder.style.fontSize = '16px';
    detectedPlaceholder.style.fontSize = '16px';
    
    if (code === 0) {
        addLog('Detection completed successfully', 'success');
        // Show the detection complete overlay
        showDetectionCompleteOverlay();
    } else {
        addLog(`Detection ended with code ${code}`, 'error');
    }
});

// Show detection complete overlay with final stats
function showDetectionCompleteOverlay() {
    document.getElementById('finalCivilians').textContent = totalCivilians;
    document.getElementById('finalSoldiers').textContent = totalSoldiers;
    document.getElementById('finalTotal').textContent = totalCivilians + totalSoldiers;
    
    document.getElementById('detectionCompleteOverlay').classList.add('show');
}

// Show detection stopped overlay with current stats
function showDetectionStoppedOverlay() {
    document.getElementById('stoppedCivilians').textContent = totalCivilians;
    document.getElementById('stoppedSoldiers').textContent = totalSoldiers;
    document.getElementById('stoppedTotal').textContent = totalCivilians + totalSoldiers;
    
    document.getElementById('detectionStoppedOverlay').classList.add('show');
}

// Hide overlays and reset for new detection
function hideDetectionCompleteOverlay() {
    document.getElementById('detectionCompleteOverlay').classList.remove('show');
}

function hideDetectionStoppedOverlay() {
    document.getElementById('detectionStoppedOverlay').classList.remove('show');
}

// Continue detection (restart with same video/stream, keeping counts)
window.continueDetection = async function() {
    hideDetectionStoppedOverlay();
    
    // Set flag to preserve cumulative counts
    isContinuing = true;
    
    addLog('Continuing detection...', 'info');
    
    // Restart detection with current settings
    await window.startDetection();
}

// Reset system for new detection
window.resetForNewDetection = function() {
    hideDetectionCompleteOverlay();
    hideDetectionStoppedOverlay();
    
    // Reset cumulative counts
    resetCumulativeCounts();
    
    // Reset video path input
    document.getElementById('videoPath').value = '';
    
    // Reset RTSP URL if applicable
    document.getElementById('rtspUrl').value = '';
    
    // Reset statistics display
    document.getElementById('civilianCount').textContent = '0';
    document.getElementById('soldierCount').textContent = '0';
    document.getElementById('personCount').textContent = '0';
    
    // Hide video display section
    document.getElementById('videoDisplaySection').style.display = 'none';
    
    // Reset placeholders
    const originalPlaceholder = document.getElementById('originalPlaceholder');
    const detectedPlaceholder = document.getElementById('detectedPlaceholder');
    originalPlaceholder.textContent = 'Waiting for video...';
    detectedPlaceholder.textContent = 'Waiting for detection...';
    originalPlaceholder.style.color = '#555';
    detectedPlaceholder.style.color = '#555';
    originalPlaceholder.style.fontSize = '13px';
    detectedPlaceholder.style.fontSize = '13px';
    
    // Reset status
    const indicator = document.getElementById('statusIndicator');
    const statusText = document.getElementById('statusText');
    indicator.className = 'status-indicator idle';
    statusText.textContent = 'Ready - Select a video or stream to begin';
    
    // Clear log and add ready message
    document.getElementById('outputArea').innerHTML = '<div class="log-entry info">System reset. Ready for new detection.</div>';
    
    addLog('System reset complete. Select a new video or stream.', 'success');
}

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
    
    // Add event listeners for buttons
    document.getElementById('videoTab').addEventListener('click', (e) => window.switchTab('video', e));
    document.getElementById('streamTab').addEventListener('click', (e) => window.switchTab('stream', e));
    document.getElementById('browseVideoBtn').addEventListener('click', window.selectVideoFile);
    document.getElementById('startBtn').addEventListener('click', window.startDetection);
    document.getElementById('stopBtn').addEventListener('click', window.stopDetection);
    document.getElementById('clearLogBtn').addEventListener('click', clearOutput);
    
    // New Detection button listeners (for both complete and stopped overlays)
    document.getElementById('newDetectionBtn').addEventListener('click', window.resetForNewDetection);
    document.getElementById('newDetectionBtnStopped').addEventListener('click', window.resetForNewDetection);
    document.getElementById('continueDetectionBtn').addEventListener('click', window.continueDetection);
    
    // Stream source change handler
    document.getElementById('streamSource').addEventListener('change', (e) => {
        document.getElementById('rtspUrlGroup').style.display = 
            e.target.value === 'rtsp' ? 'block' : 'none';
    });
});
