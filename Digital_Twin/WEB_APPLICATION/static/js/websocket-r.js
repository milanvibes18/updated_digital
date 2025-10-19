
// Enhanced WebSocket Manager
class EnhancedSocketManager {
    constructor() {
        this.socket = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.pingInterval = null;
        this.isConnected = false;
        this.eventQueue = [];
        this.pollingInterval = null;
        
        this.initConnection();
    }
    
    initConnection() {
        // Check if Socket.IO is available
        if (typeof io === 'undefined') {
            console.warn('Socket.IO not loaded, using polling fallback');
            this.setupPollingFallback();
            return;
        }
        
        try {
            this.socket = io({
                transports: ['websocket', 'polling'],
                timeout: 10000,
                forceNew: true,
                reconnection: true,
                reconnectionAttempts: this.maxReconnectAttempts,
                reconnectionDelay: this.reconnectDelay
            });
            
            this.setupEventHandlers();
            
        } catch (error) {
            console.error('Socket connection failed:', error);
            this.setupPollingFallback();
        }
    }
    
    setupEventHandlers() {
        if (!this.socket) return;
        
        this.socket.on('connect', () => {
            this.isConnected = true;
            this.reconnectAttempts = 0;
            this.updateConnectionStatus('connected');
            this.startPingPong();
            this.processEventQueue();
        });
        
        this.socket.on('disconnect', (reason) => {
            this.isConnected = false;
            this.updateConnectionStatus('disconnected');
            this.stopPingPong();
            
            if (reason === 'io server disconnect') {
                this.attemptReconnect();
            }
        });
        
        this.socket.on('connect_error', (error) => {
            this.updateConnectionStatus('error');
            this.attemptReconnect();
        });
        
        // Application events
        this.socket.on('data_update', (data) => this.handleDataUpdate(data));
        this.socket.on('alert_update', (alert) => this.handleAlertUpdate(alert));
        this.socket.on('pong', (data) => {
            const latency = Date.now() - data.timestamp;
            this.updateLatency(latency);
        });
    }
    
    setupPollingFallback() {
        this.updateConnectionStatus('polling');
        
        // Clear any existing polling interval
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
        }
        
        // Poll every 30 seconds
        this.pollingInterval = setInterval(() => {
            this.pollForUpdates();
        }, 30000);
    }
    
    pollForUpdates() {
        fetch('/api/dashboard_data')
            .then(response => {
                if (!response.ok) throw new Error('Network response was not ok');
                return response.json();
            })
            .then(data => this.handleDataUpdate(data))
            .catch(error => console.warn('Polling failed:', error));
    }
    
    attemptReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            this.setupPollingFallback();
            return;
        }
        
        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
        
        setTimeout(() => {
            if (!this.isConnected && this.socket) {
                this.socket.connect();
            }
        }, delay);
    }
    
    startPingPong() {
        this.pingInterval = setInterval(() => {
            if (this.isConnected && this.socket) {
                this.socket.emit('ping', { timestamp: Date.now() });
            }
        }, 30000);
    }
    
    stopPingPong() {
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
        }
    }
    
    emit(event, data) {
        if (this.isConnected && this.socket) {
            this.socket.emit(event, data);
        } else {
            this.eventQueue.push({ event, data, timestamp: Date.now() });
        }
    }
    
    processEventQueue() {
        while (this.eventQueue.length > 0) {
            const { event, data } = this.eventQueue.shift();
            if (this.socket) {
                this.socket.emit(event, data);
            }
        }
    }
    
    updateConnectionStatus(status) {
        const statusElement = document.getElementById('connection-status');
        const statusIndicator = document.querySelector('.status-indicator');
        
        if (statusElement) {
            const statusMap = {
                'connected': { text: 'Connected', class: 'status-online' },
                'disconnected': { text: 'Disconnected', class: 'status-warning' },
                'error': { text: 'Connection Error', class: 'status-error' },
                'polling': { text: 'Polling Mode', class: 'status-warning' }
            };
            
            const statusInfo = statusMap[status] || statusMap['disconnected'];
            statusElement.textContent = statusInfo.text;
            
            if (statusIndicator) {
                statusIndicator.className = `status-indicator ${statusInfo.class}`;
            }
        }
    }
    
    updateLatency(latency) {
        const latencyElement = document.getElementById('connection-latency');
        if (latencyElement) {
            latencyElement.textContent = `${latency}ms`;
        }
    }
    
    handleDataUpdate(data) {
        const event = new CustomEvent('socketDataUpdate', { detail: data });
        document.dispatchEvent(event);
        
        if (typeof updateDashboard === 'function') {
            updateDashboard(data);
        }
    }
    
    handleAlertUpdate(alert) {
        const event = new CustomEvent('socketAlertUpdate', { detail: alert });
        document.dispatchEvent(event);
        
        if (typeof addNewAlert === 'function') {
            addNewAlert(alert);
        }
    }
    
    cleanup() {
        this.stopPingPong();
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
        }
        if (this.socket) {
            this.socket.disconnect();
        }
    }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    window.socketManager = new EnhancedSocketManager();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.socketManager) {
        window.socketManager.cleanup();
    }
});
