// Medical Emergency Detection System - Frontend JavaScript

class DetectionSystem {
    constructor() {
        this.socket = null;
        this.isDetecting = false;
        this.detections = [];
        this.stats = {
            totalDetections: 0,
            detectionCounts: {}
        };
        
        this.initializeSocket();
        this.bindEvents();
        this.updateUI();
    }

    initializeSocket() {
        // Connect to Socket.IO server
        this.socket = io();
        
        this.socket.on('connect', () => {
            console.log('Connected to detection system');
            this.updateStatus('Connected', 'green');
        });

        this.socket.on('disconnect', () => {
            console.log('Disconnected from detection system');
            this.updateStatus('Disconnected', 'red');
        });

        this.socket.on('detection', (data) => {
            this.handleDetection(data);
        });

        this.socket.on('stats_update', (data) => {
            this.updateStats(data);
        });

        this.socket.on('fps_update', (data) => {
            this.updateFPS(data.fps);
        });

        this.socket.on('status', (data) => {
            console.log('Status:', data.message);
        });
    }

    bindEvents() {
        // Control buttons
        document.getElementById('start-btn').addEventListener('click', () => {
            this.startDetection();
        });

        document.getElementById('stop-btn').addEventListener('click', () => {
            this.stopDetection();
        });

        // Emergency modal
        document.getElementById('close-modal').addEventListener('click', () => {
            this.hideEmergencyModal();
        });
    }

    async startDetection() {
        try {
            const response = await fetch('/api/start_detection');
            const data = await response.json();
            
            if (data.status === 'success') {
                this.isDetecting = true;
                this.updateControlButtons();
                this.updateStatus('Detecting', 'blue');
                this.showNotification('Detection started successfully', 'success');
            } else {
                this.showNotification(data.message, 'error');
            }
        } catch (error) {
            console.error('Error starting detection:', error);
            this.showNotification('Failed to start detection', 'error');
        }
    }

    async stopDetection() {
        try {
            const response = await fetch('/api/stop_detection');
            const data = await response.json();
            
            if (data.status === 'success') {
                this.isDetecting = false;
                this.updateControlButtons();
                this.updateStatus('Ready', 'green');
                this.showNotification('Detection stopped', 'info');
            } else {
                this.showNotification(data.message, 'error');
            }
        } catch (error) {
            console.error('Error stopping detection:', error);
            this.showNotification('Failed to stop detection', 'error');
        }
    }

    handleDetection(detection) {
        // Add to detections array
        this.detections.unshift(detection);
        if (this.detections.length > 50) {
            this.detections = this.detections.slice(0, 50);
        }

        // Update recent detections
        this.updateRecentDetections();
        
        // Show emergency alert for critical detections
        if (detection.label === 'person_down' || detection.label === 'person_unconscious') {
            this.showEmergencyAlert(detection);
        }

        // Update detection history
        this.updateDetectionHistory();
    }

    updateStats(stats) {
        this.stats = stats;
        
        // Update statistics display
        document.getElementById('total-detections').textContent = stats.totalDetections || 0;
        document.getElementById('person-down-count').textContent = stats.detectionCounts?.person_down || 0;
        document.getElementById('injured-count').textContent = stats.detectionCounts?.person_injured || 0;
        document.getElementById('unconscious-count').textContent = stats.detectionCounts?.person_unconscious || 0;
    }

    updateFPS(fps) {
        document.getElementById('fps-display').textContent = `FPS: ${fps}`;
    }

    updateStatus(status, color) {
        const indicator = document.getElementById('status-indicator');
        const statusText = document.getElementById('status-text');
        
        // Remove all color classes
        indicator.className = 'w-3 h-3 rounded-full';
        
        // Add appropriate color
        if (color === 'green') indicator.classList.add('bg-green-500');
        else if (color === 'red') indicator.classList.add('bg-red-500');
        else if (color === 'blue') indicator.classList.add('bg-blue-500');
        else if (color === 'yellow') indicator.classList.add('bg-yellow-500');
        
        statusText.textContent = status;
    }

    updateControlButtons() {
        const startBtn = document.getElementById('start-btn');
        const stopBtn = document.getElementById('stop-btn');
        
        if (this.isDetecting) {
            startBtn.disabled = true;
            startBtn.classList.add('opacity-50', 'cursor-not-allowed');
            stopBtn.disabled = false;
            stopBtn.classList.remove('opacity-50', 'cursor-not-allowed');
        } else {
            startBtn.disabled = false;
            startBtn.classList.remove('opacity-50', 'cursor-not-allowed');
            stopBtn.disabled = true;
            stopBtn.classList.add('opacity-50', 'cursor-not-allowed');
        }
    }

    updateRecentDetections() {
        const container = document.getElementById('recent-detections');
        
        if (this.detections.length === 0) {
            container.innerHTML = '<div class="text-gray-500 text-center py-4">No detections yet</div>';
            return;
        }

        const recentDetections = this.detections.slice(0, 5);
        container.innerHTML = recentDetections.map(detection => `
            <div class="detection-card bg-gray-50 rounded-lg p-3 border-l-4 border-red-500">
                <div class="flex justify-between items-start">
                    <div>
                        <div class="font-semibold text-gray-900">${this.formatLabel(detection.label)}</div>
                        <div class="text-sm text-gray-600">${this.formatTimestamp(detection.timestamp)}</div>
                    </div>
                    <div class="text-right">
                        <div class="text-sm font-medium text-gray-900">${(detection.confidence * 100).toFixed(1)}%</div>
                        <div class="text-xs text-gray-500">confidence</div>
                    </div>
                </div>
            </div>
        `).join('');
    }

    updateDetectionHistory() {
        const container = document.getElementById('detection-history');
        
        if (this.detections.length === 0) {
            container.innerHTML = '<div class="text-gray-500 text-center py-8">No detection history available</div>';
            return;
        }

        const historyDetections = this.detections.slice(0, 12);
        container.innerHTML = historyDetections.map(detection => `
            <div class="detection-card bg-white rounded-lg shadow-md p-4 border border-gray-200">
                <div class="flex items-center justify-between mb-3">
                    <div class="flex items-center space-x-2">
                        <div class="w-3 h-3 rounded-full ${this.getDetectionColor(detection.label)}"></div>
                        <span class="font-semibold text-gray-900">${this.formatLabel(detection.label)}</span>
                    </div>
                    <span class="text-sm text-gray-500">${(detection.confidence * 100).toFixed(1)}%</span>
                </div>
                <div class="text-sm text-gray-600 mb-2">${this.formatTimestamp(detection.timestamp)}</div>
                <div class="text-xs text-gray-500">BBox: [${detection.bbox.map(b => b.toFixed(1)).join(', ')}]</div>
            </div>
        `).join('');
    }

    showEmergencyAlert(detection) {
        const modal = document.getElementById('emergency-modal');
        const details = document.getElementById('emergency-details');
        
        details.textContent = `Detected: ${this.formatLabel(detection.label)} with ${(detection.confidence * 100).toFixed(1)}% confidence at ${this.formatTimestamp(detection.timestamp)}`;
        
        modal.classList.remove('hidden');
        
        // Auto-hide after 10 seconds
        setTimeout(() => {
            this.hideEmergencyModal();
        }, 10000);
    }

    hideEmergencyModal() {
        document.getElementById('emergency-modal').classList.add('hidden');
    }

    formatLabel(label) {
        const labels = {
            'person_down': 'Person Down',
            'person_injured': 'Injured Person',
            'person_unconscious': 'Unconscious Person',
            'medical_emergency': 'Medical Emergency',
            'accident_victim': 'Accident Victim'
        };
        return labels[label] || label.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    formatTimestamp(timestamp) {
        const date = new Date(timestamp);
        return date.toLocaleTimeString();
    }

    getDetectionColor(label) {
        const colors = {
            'person_down': 'bg-red-500',
            'person_injured': 'bg-orange-500',
            'person_unconscious': 'bg-purple-500',
            'medical_emergency': 'bg-yellow-500',
            'accident_victim': 'bg-red-600'
        };
        return colors[label] || 'bg-gray-500';
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `fixed top-4 right-4 p-4 rounded-lg shadow-lg z-50 transition-all duration-300 transform translate-x-full`;
        
        const colors = {
            'success': 'bg-green-500 text-white',
            'error': 'bg-red-500 text-white',
            'info': 'bg-blue-500 text-white',
            'warning': 'bg-yellow-500 text-white'
        };
        
        notification.classList.add(colors[type] || colors.info);
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        // Animate in
        setTimeout(() => {
            notification.classList.remove('translate-x-full');
        }, 100);
        
        // Remove after 3 seconds
        setTimeout(() => {
            notification.classList.add('translate-x-full');
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 3000);
    }

    updateUI() {
        this.updateControlButtons();
        this.updateRecentDetections();
        this.updateDetectionHistory();
    }
}

// Initialize the detection system when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.detectionSystem = new DetectionSystem();
}); 