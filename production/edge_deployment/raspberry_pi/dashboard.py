#!/usr/bin/env python3
"""
Web Dashboard for Drone Detection
===================================

A visual web interface for non-technical users.
No terminal required - everything through the browser.

Features:
- Real-time detection display
- Signal strength visualization
- Detection history log
- One-click start/stop
- System status monitoring
"""

import os
import json
import time
import threading
from pathlib import Path
from datetime import datetime

from flask import Flask, render_template_string, jsonify, request

app = Flask(__name__)

# Global state
detection_state = {
    'running': False,
    'detections': [],
    'last_detection': None,
    'total_count': 0,
    'start_time': None,
    'signal_strength': 0,
}

# Dashboard HTML template (embedded for single-file deployment)
DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚁 Drone Detection System</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .pulse { animation: pulse 2s infinite; }
        @keyframes scan { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .scan { animation: scan 2s linear infinite; }
        .glass { backdrop-filter: blur(10px); background: rgba(255,255,255,0.1); }
    </style>
</head>
<body class="bg-gray-900 text-white min-h-screen">
    <!-- Header -->
    <header class="bg-gradient-to-r from-blue-600 to-purple-600 shadow-lg">
        <div class="max-w-7xl mx-auto px-4 py-4 flex justify-between items-center">
            <div class="flex items-center space-x-3">
                <div class="text-3xl">🚁</div>
                <div>
                    <h1 class="text-xl font-bold">Drone Detection System</h1>
                    <p class="text-sm opacity-80">RF-Based Real-Time Detection</p>
                </div>
            </div>
            <div id="systemStatus" class="flex items-center space-x-2">
                <span class="inline-block w-3 h-3 bg-green-400 rounded-full pulse"></span>
                <span class="text-sm">System Ready</span>
            </div>
        </div>
    </header>

    <main class="max-w-7xl mx-auto px-4 py-8">
        <!-- Control Panel -->
        <div class="grid md:grid-cols-3 gap-6 mb-8">
            <!-- Start/Stop Control -->
            <div class="bg-gray-800 rounded-xl p-6 shadow-xl">
                <h2 class="text-lg font-semibold mb-4">Control Panel</h2>
                <button id="toggleBtn" onclick="toggleDetection()" 
                    class="w-full py-4 rounded-lg text-lg font-bold transition-all
                           bg-green-500 hover:bg-green-600">
                    ▶️ Start Detection
                </button>
                <div class="mt-4 text-sm text-gray-400">
                    <p>Status: <span id="runningStatus">Stopped</span></p>
                    <p>Uptime: <span id="uptime">0:00:00</span></p>
                </div>
            </div>

            <!-- Signal Indicator -->
            <div class="bg-gray-800 rounded-xl p-6 shadow-xl">
                <h2 class="text-lg font-semibold mb-4">RF Signal</h2>
                <div class="relative h-24 flex items-center justify-center">
                    <div id="radarScan" class="w-20 h-20 border-4 border-green-500/30 rounded-full 
                                flex items-center justify-center">
                        <div class="w-12 h-12 border-4 border-green-500/50 rounded-full 
                                    flex items-center justify-center">
                            <div class="w-4 h-4 bg-green-500 rounded-full pulse"></div>
                        </div>
                    </div>
                </div>
                <div class="text-center text-sm text-gray-400">
                    Frequency: <span id="freq">2.437 GHz</span>
                </div>
            </div>

            <!-- Stats -->
            <div class="bg-gray-800 rounded-xl p-6 shadow-xl">
                <h2 class="text-lg font-semibold mb-4">Statistics</h2>
                <div class="space-y-3">
                    <div class="flex justify-between">
                        <span class="text-gray-400">Total Detections</span>
                        <span id="totalCount" class="text-2xl font-bold text-green-400">0</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Avg. Confidence</span>
                        <span id="avgConf" class="text-blue-400">--</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Last Detection</span>
                        <span id="lastTime" class="text-purple-400">--</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Latest Detection Alert -->
        <div id="alertPanel" class="hidden mb-8 bg-gradient-to-r from-red-600 to-orange-500 
                                     rounded-xl p-6 shadow-xl animate-pulse">
            <div class="flex items-center justify-between">
                <div class="flex items-center space-x-4">
                    <div class="text-5xl">⚠️</div>
                    <div>
                        <h2 class="text-2xl font-bold">DRONE DETECTED!</h2>
                        <p id="alertType" class="text-xl">DJI Phantom 4</p>
                    </div>
                </div>
                <div class="text-right">
                    <p class="text-3xl font-bold" id="alertConf">95.7%</p>
                    <p class="text-sm opacity-80">Confidence</p>
                </div>
            </div>
        </div>

        <!-- Detection History -->
        <div class="bg-gray-800 rounded-xl p-6 shadow-xl">
            <h2 class="text-lg font-semibold mb-4">Detection History</h2>
            <div class="overflow-x-auto">
                <table class="w-full">
                    <thead>
                        <tr class="text-left text-gray-400 border-b border-gray-700">
                            <th class="pb-2">Time</th>
                            <th class="pb-2">Drone Type</th>
                            <th class="pb-2">Confidence</th>
                            <th class="pb-2">Latency</th>
                        </tr>
                    </thead>
                    <tbody id="historyTable">
                        <tr class="text-gray-500">
                            <td colspan="4" class="py-4 text-center">
                                No detections yet. Click "Start Detection" to begin.
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </main>

    <!-- Footer -->
    <footer class="bg-gray-800 mt-8 py-4">
        <div class="max-w-7xl mx-auto px-4 text-center text-gray-400 text-sm">
            DroneRFB-Spectra | Powered by RF-DenseNet
        </div>
    </footer>

    <script>
        let isRunning = false;
        let pollInterval = null;

        function toggleDetection() {
            isRunning = !isRunning;
            const btn = document.getElementById('toggleBtn');
            const status = document.getElementById('runningStatus');
            const radar = document.getElementById('radarScan');
            
            if (isRunning) {
                btn.textContent = '⏹️ Stop Detection';
                btn.classList.remove('bg-green-500', 'hover:bg-green-600');
                btn.classList.add('bg-red-500', 'hover:bg-red-600');
                status.textContent = 'Running';
                radar.classList.add('scan');
                startPolling();
                
                // Notify backend
                fetch('/api/start', { method: 'POST' });
            } else {
                btn.textContent = '▶️ Start Detection';
                btn.classList.remove('bg-red-500', 'hover:bg-red-600');
                btn.classList.add('bg-green-500', 'hover:bg-green-600');
                status.textContent = 'Stopped';
                radar.classList.remove('scan');
                stopPolling();
                
                fetch('/api/stop', { method: 'POST' });
            }
        }

        function startPolling() {
            pollInterval = setInterval(fetchStatus, 500);
        }

        function stopPolling() {
            if (pollInterval) {
                clearInterval(pollInterval);
                pollInterval = null;
            }
        }

        async function fetchStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                updateUI(data);
            } catch (error) {
                console.error('Error fetching status:', error);
            }
        }

        function updateUI(data) {
            document.getElementById('totalCount').textContent = data.total_count;
            
            if (data.last_detection) {
                const det = data.last_detection;
                document.getElementById('alertPanel').classList.remove('hidden');
                document.getElementById('alertType').textContent = det.drone_type;
                document.getElementById('alertConf').textContent = (det.confidence * 100).toFixed(1) + '%';
                document.getElementById('lastTime').textContent = det.time;
                
                // Update history table
                updateHistory(data.detections);
            }
            
            if (data.start_time) {
                const elapsed = Math.floor((Date.now() / 1000) - data.start_time);
                const hours = Math.floor(elapsed / 3600);
                const mins = Math.floor((elapsed % 3600) / 60);
                const secs = elapsed % 60;
                document.getElementById('uptime').textContent = 
                    `${hours}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
            }
        }

        function updateHistory(detections) {
            const table = document.getElementById('historyTable');
            if (detections.length === 0) return;
            
            table.innerHTML = detections.slice(-10).reverse().map(d => `
                <tr class="border-b border-gray-700">
                    <td class="py-2">${d.time}</td>
                    <td class="py-2 font-semibold">${d.drone_type}</td>
                    <td class="py-2 ${d.confidence > 0.9 ? 'text-green-400' : 'text-yellow-400'}">
                        ${(d.confidence * 100).toFixed(1)}%
                    </td>
                    <td class="py-2 text-gray-400">${d.latency_ms.toFixed(1)}ms</td>
                </tr>
            `).join('');
        }
    </script>
</body>
</html>
'''


@app.route('/')
def dashboard():
    """Serve the main dashboard."""
    return render_template_string(DASHBOARD_HTML)


@app.route('/api/start', methods=['POST'])
def start_detection():
    """Start detection process."""
    detection_state['running'] = True
    detection_state['start_time'] = time.time()
    detection_state['detections'] = []
    detection_state['total_count'] = 0
    
    # Start background detection thread
    thread = threading.Thread(target=background_detection, daemon=True)
    thread.start()
    
    return jsonify({'status': 'started'})


@app.route('/api/stop', methods=['POST'])
def stop_detection():
    """Stop detection process."""
    detection_state['running'] = False
    return jsonify({'status': 'stopped'})


@app.route('/api/status')
def get_status():
    """Get current detection status."""
    return jsonify(detection_state)


@app.route('/health')
def health_check():
    """Health check endpoint for Docker."""
    return jsonify({'status': 'healthy', 'timestamp': time.time()})


def background_detection():
    """
    Background detection loop.
    In production, this would use real RTL-SDR data.
    For demo, it simulates detections.
    """
    import random
    drone_types = ['DJI Phantom 4', 'DJI Mavic Air', 'FrSky Taranis', 
                   'Futaba T8J', 'RadioLink AT9S', 'Background Noise']
    
    while detection_state['running']:
        # Simulate detection (replace with real RTL-SDR + model inference)
        confidence = random.random()
        
        if confidence > 0.5:  # Detection threshold
            drone_type = random.choice(drone_types[:-1])
        else:
            drone_type = 'Background Noise'
            confidence = 0.1 + random.random() * 0.3
        
        detection = {
            'time': datetime.now().strftime('%H:%M:%S'),
            'drone_type': drone_type,
            'confidence': confidence,
            'latency_ms': 4 + random.random() * 3,
        }
        
        if confidence > 0.7:  # Only log high-confidence detections
            detection_state['detections'].append(detection)
            detection_state['last_detection'] = detection
            detection_state['total_count'] += 1
        
        time.sleep(0.5)  # 2 scans per second


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Drone Detection Dashboard')
    parser.add_argument('--host', default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=8080, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚁 Drone Detection Dashboard")
    print("=" * 60)
    print(f"\n🌐 Open your browser and go to:")
    print(f"   http://localhost:{args.port}")
    print(f"   http://<your-ip>:{args.port}")
    print("\n📡 Dashboard ready for drone detection!")
    print("=" * 60 + "\n")
    
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
