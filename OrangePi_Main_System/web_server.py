import os
import logging
from flask import Flask, jsonify, render_template_string, request

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ASV HMI</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { background-color: #0d0d0d; color: #00ffcc; font-family: 'Courier New', Courier, monospace; text-align: center; }
        h1 { color: #ffffff; text-shadow: 0 0 10px #00ffcc; margin-bottom: 20px; }
        .dashboard { display: flex; justify-content: center; gap: 40px; align-items: flex-start; flex-wrap: wrap; }
        .column { display: flex; flex-direction: column; gap: 20px; }
        .card { background: #1a1a1a; padding: 20px; border-radius: 10px; border: 1px solid #333; box-shadow: 0 4px 8px rgba(0,0,0,0.5); min-width: 250px; display: flex; flex-direction: column; align-items: center;}
        .card h2 { color: #ffcc00; margin-top: 0; border-bottom: 1px solid #444; padding-bottom: 10px;}
        .val { color: #fff; font-weight: bold; font-size: 1.2em;}
        #radar-box { position: relative; width: 500px; height: 500px; background-color: #001100; border: 2px solid #00ff00; border-radius: 5px; box-shadow: 0 0 15px #00ff00; margin: 0 auto; }
        canvas { position: absolute; top: 0; left: 0; }
        .btn { background: #111; color: #fff; border: 2px solid #555; padding: 15px 20px; font-size: 1.1em; font-weight: bold; border-radius: 5px; cursor: pointer; transition: 0.2s; width: 100%; margin-bottom: 15px; text-transform: uppercase;}
        .btn:hover { background: #222; }
        .btn:active { transform: scale(0.98); }
        .btn-run { border-color: #00ff00; color: #00ff00; }
        .btn-stop { border-color: #ff0000; color: #ff0000; }
        .btn-rtl { border-color: #ffaa00; color: #ffaa00; }
        .status-armed { color: #00ff00; font-weight: bold; text-shadow: 0 0 8px #00ff00; }
        .status-disarmed { color: #ff0000; font-weight: bold; text-shadow: 0 0 8px #ff0000; }
        .chart-container { position: relative; width: 600px; height: 400px; margin: 0 auto; }
    </style>
</head>
<body>
    <h1>ASV HMI COMMAND CENTER</h1>
    <div class="dashboard">
        <div class="column">
            <div class="card">
                <h2>Control Panel</h2>
                <button class="btn btn-run" onclick="sendCommand('RUN')">▶ Start Mission</button>
                <button class="btn btn-stop" onclick="sendCommand('STOP')">🛑 Emergency Stop</button>
                <button class="btn btn-rtl" onclick="sendCommand('RTL')">🏠 Return to Home</button>
                <p style="margin-top: 15px;">Engine: <span id="engine_status" class="status-disarmed">DISARMED</span></p>
            </div>
            <div class="card">
                <h2>State panel</h2>
                <p>State: <span class="val" id="state" style="color: #ff00ff;">-</span></p>
                <p>X: <span class="val" id="b_x">0.0</span> | Y: <span class="val" id="b_y">0.0</span> m</p>
                <p>Target Yaw: <span class="val" id="t_yaw" style="color: #ff00ff;">0.0</span>°</p>
                <p>Current Yaw: <span class="val" id="b_yaw" style="color: #00ffcc;">0.0</span>°</p>
            </div>
        </div>
        
        <div class="card">
            <h2>MAP</h2>
            <div id="radar-box"><canvas id="radar" width="500" height="500"></canvas></div>
        </div>

        <div class="card">
            <h2>YAW RESPONSE</h2>
            <div class="chart-container">
                <canvas id="yawChart"></canvas>
            </div>
        </div>
    </div>

    <script>
        // --- RADAR SETUP ---
        const canvas = document.getElementById('radar');
        const ctx = canvas.getContext('2d');
        
        const MAP_MIN_X = -1.0;
        const MAP_MAX_X = 5.0;
        const MAP_MAX_Y = 1.0;
        const MAP_MIN_Y = -5.0;
        const CANVAS_SIZE = 500;
        
        function toPixelX(realX) {
            return ((realX - MAP_MIN_X) / (MAP_MAX_X - MAP_MIN_X)) * CANVAS_SIZE;
        }
        function toPixelY(realY) {
            return ((MAP_MAX_Y - realY) / (MAP_MAX_Y - MAP_MIN_Y)) * CANVAS_SIZE;
        }
        
        // --- CHART SETUP ---
        const ctxChart = document.getElementById('yawChart').getContext('2d');
        const MAX_POINTS = 50;
        const chartLabels = new Array(MAX_POINTS).fill('');
        const dataCurrentYaw = new Array(MAX_POINTS).fill(0);
        const dataTargetYaw = new Array(MAX_POINTS).fill(0);
        
        const yawChart = new Chart(ctxChart, {
            type: 'line',
            data: {
                labels: chartLabels,
                datasets: [
                    { label: 'Current Yaw (deg)', borderColor: '#00ffcc', data: dataCurrentYaw, fill: false, tension: 0.2, pointRadius: 0, borderWidth: 2 },
                    { label: 'Target Yaw (deg)', borderColor: '#ff00ff', data: dataTargetYaw, fill: false, tension: 0.2, pointRadius: 0, borderWidth: 2, borderDash: [5, 5] }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false, animation: false,
                scales: { 
                    // [SỬA LỖI]: Mở rộng trục y từ -360 đến 360
                    y: { min: -360, max: 360, grid: { color: '#333' } }, 
                    x: { grid: { display: false } }
                },
                plugins: { legend: { labels: { color: '#fff' } } }
            }
        });

        function sendCommand(cmd) { fetch('/api/cmd', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ command: cmd }) }); }
        
        if (window.radarInterval) clearInterval(window.radarInterval);

        function updateRadar() {
            fetch('/api/data').then(r => r.json()).then(data => {
                
                // [SỬA LỖI 1]: Dùng trực tiếp data.usv_yaw nguyên thủy từ IMU để tránh bị ngược dấu âm do tính toán qua boat_yaw
                let currentYawDeg = data.usv_yaw;
                let targetYawDeg = data.target_yaw;

                // [SỬA LỖI 2 - DÍNH PHA]: Kéo góc Target về gần với góc Current nhất để đồ thị không bị lệch 360 độ
                // Nếu Current = -10, Target = 350 -> Thuật toán này sẽ ép Target thành -10 để đồ thị vẽ đè lên nhau.
                while (targetYawDeg - currentYawDeg > 180) targetYawDeg -= 360;
                while (targetYawDeg - currentYawDeg < -180) targetYawDeg += 360;

                // Update Texts
                document.getElementById('state').innerText = data.state;
                document.getElementById('b_x').innerText = data.boat_x.toFixed(2);
                document.getElementById('b_y').innerText = data.boat_y.toFixed(2);
                document.getElementById('b_yaw').innerText = currentYawDeg.toFixed(1);
                document.getElementById('t_yaw').innerText = targetYawDeg.toFixed(1);
                
                let engineEl = document.getElementById('engine_status');
                engineEl.innerText = data.is_running ? "ARMED (LIVE)" : "DISARMED";
                engineEl.className = data.is_running ? "status-armed" : "status-disarmed";

                // Update Chart
                yawChart.data.datasets[0].data.push(currentYawDeg);
                yawChart.data.datasets[0].data.shift();
                yawChart.data.datasets[1].data.push(targetYawDeg);
                yawChart.data.datasets[1].data.shift();
                yawChart.update();

                // Draw Radar
                ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE); 
                
                ctx.strokeStyle = '#004400'; ctx.lineWidth = 1;
                let stepX = CANVAS_SIZE / (MAP_MAX_X - MAP_MIN_X);
                let stepY = CANVAS_SIZE / (MAP_MAX_Y - MAP_MIN_Y);
                for(let i=0; i<=CANVAS_SIZE; i+=stepX) { ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, CANVAS_SIZE); ctx.stroke(); }
                for(let i=0; i<=CANVAS_SIZE; i+=stepY) { ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(CANVAS_SIZE, i); ctx.stroke(); }
                
                ctx.fillStyle = 'rgba(255, 255, 0, 0.3)'; 
                let homePx_startX = toPixelX(0.0);
                let homePx_startY = toPixelY(0.0);
                let homeWidthPx = toPixelX(0.5) - toPixelX(0.0);
                let homeHeightPx = toPixelY(-0.5) - toPixelY(0.0);
                ctx.fillRect(homePx_startX, homePx_startY, homeWidthPx, homeHeightPx);

                ctx.fillStyle = '#ff0000';
                data.active_trash.forEach(t => {
                    let pxX = toPixelX(t.x);
                    let pxY = toPixelY(t.y);
                    ctx.beginPath(); ctx.arc(pxX, pxY, 6, 0, Math.PI*2); ctx.fill();
                    ctx.shadowBlur = 10; ctx.shadowColor = "red";
                });
                ctx.shadowBlur = 0;
                
                // Vẽ Thuyền
                let boatPxX = toPixelX(data.boat_x);
                let boatPxY = toPixelY(data.boat_y);
                ctx.save();
                ctx.translate(boatPxX, boatPxY);
                // Radar sử dụng data.boat_yaw để vẽ đúng chiều định vị
                ctx.rotate(-data.boat_yaw); 
                ctx.fillStyle = '#0088ff'; ctx.beginPath(); ctx.moveTo(15, 0); ctx.lineTo(-10, 10); ctx.lineTo(-5, 0); ctx.lineTo(-10, -10); ctx.fill();
                ctx.restore();
            });
        }
        window.radarInterval = setInterval(updateRadar, 33);
    </script>
</body>
</html>
"""

def web_server_process(shared_dict):
    try: os.sched_setaffinity(0, {7})
    except: pass
    
    app = Flask(__name__)
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    @app.route('/')
    def index(): return render_template_string(HTML_TEMPLATE)

    @app.route('/api/data')
    def get_data(): return jsonify(dict(shared_dict))

    @app.route('/api/cmd', methods=['POST'])
    def handle_command():
        cmd = request.get_json().get('command', '')
        shared_dict['web_cmd'] = cmd
        return jsonify({"status": "ok"})

    print("--> [WEB] Radar Server Active! Access: http://<IP>:8080")
    app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)