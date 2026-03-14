"""
Web Dashboard for Crowd Prediction System

Provides a browser-based view with:
- Live video with overlays
- Current risk and stampede probability
- Emergency call workflow with 10s cancel window
"""

import os
import time
import threading
from collections import deque
from typing import Optional, Dict, Any

import cv2
from flask import Flask, Response, jsonify, request, render_template_string

from crowd_prediction_system import CrowdPredictionSystem
from risk_analyzer import RiskReport, RiskLevel


#VIDEO_PATH = os.environ.get("CROWD_VIDEO_PATH", r"crowd.mp4")
VIDEO_PATH = os.environ.get("CROWD_VIDEO_PATH", r"crowd 2.mp4")
#VIDEO_PATH = os.environ.get("CROWD_VIDEO_PATH", r"crowd3.webm")
MODEL_PATH = os.environ.get("CROWD_MODEL_PATH", "crowd_predictor_model.pth")


app = Flask(__name__)


def _draw_stampede_zones_view(
    raw_bgr,
    risk_report: Optional[RiskReport],
    emergency_notify: bool,
    frame_count: int,
) -> Any:
    """
    Raw scene + marked zones where crush/stampede risk is concentrated
    (density hotspots). While emergency services are being notified, adds
    a dispatch banner and stronger markers.
    """
    out = raw_bgr.copy()
    h, w = out.shape[:2]
    scale = max(w, h) / 1000.0

    if risk_report and risk_report.hotspots:
        for i, (hx, hy, hrisk) in enumerate(risk_report.hotspots):
            cx, cy = int(hx), int(hy)
            cx = max(0, min(w - 1, cx))
            cy = max(0, min(h - 1, cy))
            r = int(35 + 80 * float(hrisk))
            thickness = 4 if emergency_notify else 2
            # Outer warning ring (BGR red)
            cv2.circle(out, (cx, cy), r, (0, 0, 255), thickness)
            cv2.circle(out, (cx, cy), max(8, r // 4), (0, 165, 255), -1)
            label = f"ZONE {i + 1}  {float(hrisk) * 100:.0f}%"
            cv2.putText(
                out,
                label,
                (cx - min(120, cx), max(22, cy - r - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55 * scale,
                (255, 255, 255),
                max(1, int(2 * scale)),
                cv2.LINE_AA,
            )
            cv2.putText(
                out,
                "STAMPEDE RISK",
                (cx - min(100, cx), min(h - 8, cy + r + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45 * scale,
                (200, 200, 255),
                1,
                cv2.LINE_AA,
            )

    # Predicted crowding at +2s from LSTM paths (optional secondary marks)
    if risk_report and emergency_notify:
        bar = out.copy()
        cv2.rectangle(bar, (0, 0), (w, 52), (0, 0, 160), -1)
        cv2.addWeighted(bar, 0.85, out, 0.15, 0, out)
        cv2.putText(
            out,
            "EMERGENCY SERVICES NOTIFIED — STAMPEDE-PRONE ZONES BELOW",
            (12, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65 * scale,
            (255, 255, 255),
            max(1, int(2 * scale)),
            cv2.LINE_AA,
        )
    elif risk_report and risk_report.hotspots:
        cv2.putText(
            out,
            "Stampede-zone map (density / convergence)",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5 * scale,
            (220, 220, 255),
            1,
            cv2.LINE_AA,
        )

    return out


class SystemRunner:
    """
    Runs the CrowdPredictionSystem in a background thread and exposes
    latest frame and risk report for the web dashboard.
    """

    def __init__(self, video_path: str, model_path: Optional[str]):
        self.video_path = video_path
        self.model_path = model_path
        self.system = CrowdPredictionSystem(
            video_path=video_path,
            model_path=model_path,
            prediction_enabled=True,
        )

        self._lock = threading.Lock()
        self._latest_views: Dict[str, Any] = {}  # different visual views
        self._latest_risk: Optional[RiskReport] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._started_at = time.time()
        self._frame_count = 0
        self._last_tracked = 0
        self._loop_durations: deque = deque(maxlen=30)  # for smoothed FPS
        self._last_pipeline_fps = 0.0

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _loop(self):
        cap = self.system.cap  # already opened in constructor

        # Run as fast as processing allows, with only a tiny sleep to yield
        # to the event loop. This removes artificial throttling so the
        # effective frame rate is limited primarily by model processing time.
        while self._running:
            loop_t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Preserve raw frame (no overlays)
            raw_frame = frame.copy()

            data = self.system.process_frame(frame)
            main_display = self.system.draw_main_frame(frame, data)

            # Create additional visualizations (dot matrix, prediction panel)
            dot_matrix = self.system.dot_visualizer.create_visualization(
                self.system.tracker, frame.shape
            )

            if self.system.show_predictions and data["predictions"]:
                pred_display = self.system.pred_visualizer.create_visualization(
                    data["current_state"],
                    data["predictions"],
                    data["risk_report"],
                    data["trails"],
                    self.system.prediction_horizon,
                    show_heatmap=self.system.show_heatmap,
                )
            else:
                pred_display = self.system.pred_visualizer.create_visualization(
                    data["current_state"],
                    None,
                    data["risk_report"],
                    data["trails"],
                    show_heatmap=self.system.show_heatmap,
                )

            risk_rep = data.get("risk_report")
            notify = bool(EMERGENCY_STATE.get("active_call"))
            stampede_view = _draw_stampede_zones_view(
                raw_frame, risk_rep, notify, self._frame_count + 1
            )

            dt = time.perf_counter() - loop_t0
            self._loop_durations.append(dt)
            if self._loop_durations:
                avg_dt = sum(self._loop_durations) / len(self._loop_durations)
                self._last_pipeline_fps = 1.0 / max(avg_dt, 1e-6)

            with self._lock:
                self._frame_count += 1
                self._last_tracked = len(data.get("current_state") or {})
                self._latest_views = {
                    "raw": raw_frame,
                    "main": main_display,
                    "dot": dot_matrix,
                    "prediction": pred_display,
                    "stampede": stampede_view,
                }
                self._latest_risk = data.get("risk_report")

            # Small sleep so we don't starve the event loop / CPU completely
            time.sleep(0.01)

        cap.release()

    def get_latest_frame(self, view: str = "main"):
        with self._lock:
            frame = self._latest_views.get(view)
            if frame is None:
                frame = self._latest_views.get("main")
            return None if frame is None else frame.copy()

    def get_latest_risk(self) -> Optional[RiskReport]:
        with self._lock:
            return self._latest_risk

    def get_system_snapshot(self) -> Dict[str, Any]:
        """Operational metrics for a realistic status panel."""
        with self._lock:
            fc = self._frame_count
            fps = float(self._last_pipeline_fps)
            tracked = self._last_tracked
        uptime = time.time() - self._started_at
        src = os.path.basename(self.video_path) or self.video_path
        w = getattr(self.system, "frame_width", 0)
        h = getattr(self.system, "frame_height", 0)
        model_ok = self.model_path is not None
        pred_every = getattr(self.system, "prediction_interval", 15)
        # Pipeline health: FPS and frames processed
        if fps >= 1.0:
            health = "Nominal"
            health_detail = "Processing within SLA"
        elif fps >= 0.3:
            health = "Degraded"
            health_detail = "Heavy load — latency elevated"
        else:
            health = "Limited"
            health_detail = "Pipeline backlogged"
        return {
            "pipeline_health": health,
            "pipeline_detail": health_detail,
            "pipeline_fps": round(fps, 2),
            "frames_processed": fc,
            "uptime_sec": round(uptime, 1),
            "ingest_source": src,
            "ingest_resolution": f"{w}×{h}" if w and h else "—",
            "source_fps_nominal": int(getattr(self.system, "fps", 30) or 30),
            "trackers_active": tracked,
            "model_profile": "LSTM + risk fusion" if model_ok else "Kinematic extrapolation",
            "model_loaded": model_ok,
            "prediction_cadence_frames": pred_every,
        }


def _serialize_risk(report: Optional[RiskReport]) -> Dict[str, Any]:
    if report is None:
        return {
            "available": False,
            "risk_level": "unknown",
            "density_risk": 0.0,
            "velocity_risk": 0.0,
            "convergence_risk": 0.0,
            "stampede_probability": 0.0,
            "overall_risk": 0.0,
            "timestamp": None,
            "should_auto_call": False,
        }

    # Define when the system considers a stampede likely enough to suggest calling.
    # Requirement: prediction probability must be above 75%.
    stampede_flag = bool(float(report.stampede_probability) >= 0.75)

    return {
        "available": True,
        "risk_level": report.risk_level.value,
        "density_risk": float(report.density_risk),
        "velocity_risk": float(report.velocity_risk),
        "convergence_risk": float(report.convergence_risk),
        "stampede_probability": float(report.stampede_probability),
        "overall_risk": float(report.overall_risk),
        "timestamp": float(report.timestamp),
        "should_auto_call": bool(stampede_flag),
    }


model_to_use = MODEL_PATH if os.path.exists(MODEL_PATH) else None
system_runner: Optional[SystemRunner] = SystemRunner(VIDEO_PATH, model_to_use)
system_runner.start()


def _frame_generator(view: str = "main"):
    """MJPEG stream generator for latest frames for a given view."""
    while True:
        if system_runner is None:
            time.sleep(0.1)
            continue

        frame = system_runner.get_latest_frame(view=view)
        if frame is None:
            time.sleep(0.05)
            continue

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            time.sleep(0.05)
            continue

        jpg_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    """Live video feed with selectable views."""
    view = request.args.get("view", "main")
    return Response(
        _frame_generator(view=view),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/status")
def status():
    """Current risk, stampede status, and system operations snapshot."""
    if system_runner is None:
        return jsonify({"available": False}), 503

    report = system_runner.get_latest_risk()
    payload = _serialize_risk(report)
    payload["system"] = system_runner.get_system_snapshot()
    return jsonify(payload)


EMERGENCY_STATE = {
    "last_call_time": None,
    "last_cancel_time": None,
    "active_call": False,
    "zones_at_dispatch": [],  # [[x,y,risk], ...] when call placed
}


@app.route("/emergency_call", methods=["POST"])
def emergency_call():
    """
    Endpoint invoked when the 10s timer completes or the user presses
    'Call Now'. In real deployments this would integrate with an
    external dispatch or telephony system.
    """
    EMERGENCY_STATE["last_call_time"] = time.time()
    EMERGENCY_STATE["active_call"] = True
    zones = []
    if system_runner:
        rep = system_runner.get_latest_risk()
        if rep and rep.hotspots:
            zones = [
                [float(x), float(y), float(r)] for x, y, r in rep.hotspots
            ]
    EMERGENCY_STATE["zones_at_dispatch"] = zones
    return jsonify({
        "status": "calling",
        "message": "Emergency services have been notified.",
        "stampede_zones": zones,
        "open_stampede_view": True,
    })


@app.route("/cancel_emergency", methods=["POST"])
def cancel_emergency():
    """User cancelled the pending emergency call before the timer completed."""
    EMERGENCY_STATE["last_cancel_time"] = time.time()
    EMERGENCY_STATE["active_call"] = False
    return jsonify({"status": "cancelled"})


INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Percepta – Crowd Safety Dashboard</title>
  <style>
    body {
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #060712;
      color: #f4f4f7;
      display: flex;
      flex-direction: column;
      min-height: 100vh;
    }
    header {
      padding: 16px 24px;
      background: linear-gradient(90deg, #111827, #0b1120);
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid #1f2937;
    }
    header h1 {
      margin: 0;
      font-size: 1.3rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #e5e7eb;
    }
    .badge {
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.75rem;
      background: #1f2937;
      color: #9ca3af;
      border: 1px solid #374151;
    }
    main {
      flex: 1;
      display: grid;
      grid-template-columns: 3fr 2fr;
      gap: 18px;
      padding: 18px 20px 24px;
    }
    .card {
      background: radial-gradient(circle at top left, #111827, #020617);
      border-radius: 14px;
      border: 1px solid #111827;
      box-shadow: 0 20px 40px rgba(0,0,0,0.7);
      padding: 16px 16px 18px;
      position: relative;
      overflow: hidden;
    }
    .card h2 {
      margin: 0 0 8px;
      font-size: 0.95rem;
      text-transform: uppercase;
      letter-spacing: 0.18em;
      color: #9ca3af;
    }
    .video-wrapper {
      margin-top: 8px;
      border-radius: 12px;
      overflow: hidden;
      border: 1px solid #111827;
      background: #020617;
      display: flex;
      justify-content: center;
      align-items: center;
      max-height: 560px;
    }
    .video-wrapper img {
      max-width: 100%;
      height: auto;
      display: block;
    }
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 10px;
    }
    .stat {
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid #1f2937;
      background: linear-gradient(135deg, #020617, #030712);
    }
    .stat-label {
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      color: #6b7280;
      margin-bottom: 6px;
    }
    .stat-value {
      font-size: 1rem;
      font-weight: 600;
    }
    .stat-sub {
      font-size: 0.75rem;
      color: #9ca3af;
      margin-top: 2px;
    }
    .risk-chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.75rem;
      font-weight: 500;
      background: #111827;
      border: 1px solid #1f2937;
    }
    .risk-dot {
      width: 8px;
      height: 8px;
      border-radius: 999px;
    }
    .risk-dot.safe { background: #22c55e; }
    .risk-dot.caution { background: #eab308; }
    .risk-dot.warning { background: #f97316; }
    .risk-dot.danger { background: #ef4444; }
    .risk-dot.critical { background: #b91c1c; }
    .risk-text-safe { color: #22c55e; }
    .risk-text-caution { color: #eab308; }
    .risk-text-warning { color: #f97316; }
    .risk-text-danger { color: #ef4444; }
    .risk-text-critical { color: #fca5a5; }
    .emergency-panel {
      margin-top: 14px;
      padding: 12px 12px 13px;
      border-radius: 12px;
      border: 1px solid rgba(220, 38, 38, 0.4);
      background: radial-gradient(circle at top left, rgba(30, 64, 175, 0.5), rgba(15,23,42,0.9));
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .emergency-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
    }
    .emergency-title {
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      color: #fee2e2;
    }
    .emergency-btn-primary {
      padding: 8px 14px;
      border-radius: 999px;
      border: none;
      cursor: pointer;
      font-size: 0.8rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      background: linear-gradient(135deg, #dc2626, #f97316);
      color: white;
      box-shadow: 0 12px 30px rgba(220,38,38,0.6);
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .emergency-btn-primary:disabled {
      opacity: 0.6;
      cursor: default;
      box-shadow: none;
    }
    .emergency-icon {
      width: 18px;
      height: 18px;
      border-radius: 999px;
      border: 2px solid rgba(248, 250, 252, 0.6);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 11px;
    }
    .emergency-body {
      font-size: 0.8rem;
      color: #e5e7eb;
      line-height: 1.4;
    }
    .emergency-footer {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 6px;
      gap: 10px;
      font-size: 0.78rem;
    }
    .timer-pill {
      padding: 4px 10px;
      border-radius: 999px;
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid rgba(148, 163, 184, 0.5);
      color: #e5e7eb;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .timer-dot {
      width: 6px;
      height: 6px;
      border-radius: 999px;
      background: #22c55e;
    }
    .cancel-link {
      border: none;
      background: transparent;
      padding: 0;
      color: #e5e7eb;
      text-decoration: underline;
      cursor: pointer;
    }
    footer {
      padding: 10px 22px 14px;
      font-size: 0.7rem;
      color: #6b7280;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-top: 1px solid #020617;
      background: #020617;
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Percepta</h1>
      <div style="font-size: 0.75rem; color: #6b7280; margin-top: 2px;">
        Real-time crowd risk and emergency escalation
      </div>
    </div>
    <div>
      <span class="badge">Web Dashboard · Local</span>
    </div>
  </header>

  <main>
    <section class="card">
      <h2>Live Monitoring</h2>
      <div style="display:flex; gap:8px; margin-top:4px; margin-bottom:6px; font-size:0.78rem;">
        <button id="view-raw" style="padding:4px 10px; border-radius:999px; border:1px solid #1f2937; background:#020617; color:#9ca3af; cursor:pointer;">Raw Video</button>
        <button id="view-main" style="padding:4px 10px; border-radius:999px; border:1px solid #4b5563; background:#111827; color:#e5e7eb; cursor:pointer;">Real Recording</button>
        <button id="view-dot" style="padding:4px 10px; border-radius:999px; border:1px solid #1f2937; background:#020617; color:#9ca3af; cursor:pointer;">Dot Matrix</button>
        <button id="view-prediction" style="padding:4px 10px; border-radius:999px; border:1px solid #1f2937; background:#020617; color:#9ca3af; cursor:pointer;">Prediction & Risk</button>
        <button id="view-stampede" style="padding:4px 10px; border-radius:999px; border:1px solid #7f1d1d; background:#1c1917; color:#fecaca; cursor:pointer;" title="Hotspots where stampede risk is highest">Stampede zones</button>
      </div>
      <div class="video-wrapper">
        <img id="video-feed" src="/video_feed?view=main" alt="Live crowd monitoring" />
      </div>
    </section>

    <section class="card">
      <h2>Risk & Emergency</h2>
      <div class="stats-grid">
        <div class="stat">
          <div class="stat-label">Overall Risk</div>
          <div class="stat-value">
            <span id="overall-risk-value">--%</span>
          </div>
          <div class="stat-sub">
            <span id="risk-level-chip" class="risk-chip">
              <span class="risk-dot" id="risk-level-dot"></span>
              <span id="risk-level-text">Waiting for data…</span>
            </span>
          </div>
        </div>
        <div class="stat">
          <div class="stat-label">Stampede Probability</div>
          <div class="stat-value" id="stampede-value">--%</div>
          <div class="stat-sub" id="stampede-sub">Model evaluating movement patterns…</div>
        </div>
        <div class="stat">
          <div class="stat-label">Density / Velocity</div>
          <div class="stat-value">
            <span id="density-risk">--%</span>
            <span style="color:#4b5563;"> · </span>
            <span id="velocity-risk">--%</span>
          </div>
          <div class="stat-sub">Crowd concentration & sudden movement</div>
        </div>
        <div class="stat" style="grid-column: 1 / -1;">
          <div class="stat-label">Operations · Percepta Core</div>
          <div class="stat-value" id="system-status" style="font-size:0.95rem;">Initializing pipeline…</div>
          <div class="stat-sub" id="system-timestamp" style="margin-top:6px; line-height:1.45;">
            <span id="sys-line1">—</span><br/>
            <span id="sys-line2" style="color:#9ca3af;">—</span>
          </div>
        </div>
      </div>

      <div class="emergency-panel" id="emergency-panel">
        <div class="emergency-header">
          <div class="emergency-title">Emergency Call</div>
          <button class="emergency-btn-primary" id="manual-emergency-btn">
            <span class="emergency-icon">!</span>
            Call Emergency Now
          </button>
        </div>
        <div class="emergency-body" id="emergency-message">
          When the system predicts a potential stampede, it will automatically prepare to call emergency services.
          Operators have a 10 second window to cancel before the call is placed.
        </div>
        <div class="emergency-footer">
          <div class="timer-pill" id="timer-pill" style="display:none;">
            <span class="timer-dot"></span>
            <span id="timer-text">Calling in 10s…</span>
          </div>
          <div>
            <button class="cancel-link" id="cancel-call-btn" style="display:none;">Cancel call</button>
          </div>
        </div>
      </div>
    </section>
  </main>

  <footer>
    <span>Predictions are advisory. Always follow on-site safety protocols.</span>
    <span id="auto-trigger-indicator"></span>
  </footer>

  <script>
    const videoEl = document.getElementById("video-feed");
    const btnRaw = document.getElementById("view-raw");
    const btnMain = document.getElementById("view-main");
    const btnDot = document.getElementById("view-dot");
    const btnPrediction = document.getElementById("view-prediction");
    const btnStampede = document.getElementById("view-stampede");
    function setActiveViewButton(active) {
      const buttons = [
        { el: btnRaw, id: "raw" },
        { el: btnMain, id: "main" },
        { el: btnDot, id: "dot" },
        { el: btnPrediction, id: "prediction" },
        { el: btnStampede, id: "stampede" },
      ];
      buttons.forEach(({ el, id }) => {
        if (id === active) {
          el.style.background = "#111827";
          el.style.color = "#e5e7eb";
          el.style.borderColor = "#4b5563";
        } else {
          el.style.background = "#020617";
          el.style.color = "#9ca3af";
          el.style.borderColor = "#1f2937";
        }
      });
      if (btnStampede && active !== "stampede") {
        btnStampede.style.background = "#1c1917";
        btnStampede.style.color = "#fecaca";
        btnStampede.style.borderColor = "#7f1d1d";
      } else if (btnStampede && active === "stampede") {
        btnStampede.style.background = "#450a0a";
        btnStampede.style.color = "#fff";
        btnStampede.style.borderColor = "#ef4444";
      }
    }

    btnRaw.addEventListener("click", () => {
      videoEl.src = "/video_feed?view=raw";
      setActiveViewButton("raw");
    });
    btnMain.addEventListener("click", () => {
      videoEl.src = "/video_feed?view=main";
      setActiveViewButton("main");
    });
    btnDot.addEventListener("click", () => {
      videoEl.src = "/video_feed?view=dot";
      setActiveViewButton("dot");
    });
    btnPrediction.addEventListener("click", () => {
      videoEl.src = "/video_feed?view=prediction";
      setActiveViewButton("prediction");
    });
    btnStampede.addEventListener("click", () => {
      videoEl.src = "/video_feed?view=stampede&_=" + Date.now();
      setActiveViewButton("stampede");
    });

    const statusUrl = "/status";
    const emergencyUrl = "/emergency_call";
    const cancelUrl = "/cancel_emergency";

    const overallRiskEl = document.getElementById("overall-risk-value");
    const riskLevelTextEl = document.getElementById("risk-level-text");
    const riskLevelDotEl = document.getElementById("risk-level-dot");
    const riskLevelChipEl = document.getElementById("risk-level-chip");
    const stampedeValueEl = document.getElementById("stampede-value");
    const stampedeSubEl = document.getElementById("stampede-sub");
    const densityRiskEl = document.getElementById("density-risk");
    const velocityRiskEl = document.getElementById("velocity-risk");
    const systemStatusEl = document.getElementById("system-status");
    const systemTimestampEl = document.getElementById("system-timestamp");
    const sysLine1El = document.getElementById("sys-line1");
    const sysLine2El = document.getElementById("sys-line2");
    const autoTriggerIndicatorEl = document.getElementById("auto-trigger-indicator");

    const manualBtn = document.getElementById("manual-emergency-btn");
    const emergencyMessageEl = document.getElementById("emergency-message");
    const timerPillEl = document.getElementById("timer-pill");
    const timerTextEl = document.getElementById("timer-text");
    const cancelCallBtn = document.getElementById("cancel-call-btn");

    let countdownInterval = null;
    let pendingAutoTrigger = false;
    let callInProgress = false;

    function setRiskLevel(level) {
      const classes = ["safe", "caution", "warning", "danger", "critical"];
      classes.forEach(c => {
        riskLevelDotEl.classList.remove(c);
        riskLevelChipEl.classList.remove("risk-text-" + c);
      });
      if (classes.includes(level)) {
        riskLevelDotEl.classList.add(level);
        riskLevelChipEl.classList.add("risk-text-" + level);
      }
    }

    function formatPct(value) {
      if (value == null) return "--%";
      return (value * 100).toFixed(0) + "%";
    }

    function formatTime(ts) {
      if (!ts && ts !== 0) return "—";
      const d = new Date(ts * 1000);
      return d.toLocaleTimeString();
    }

    function formatUptime(sec) {
      if (sec == null || sec < 0) return "0s";
      const h = Math.floor(sec / 3600);
      const m = Math.floor((sec % 3600) / 60);
      const s = Math.floor(sec % 60);
      if (h > 0) return h + "h " + m + "m " + s + "s";
      if (m > 0) return m + "m " + s + "s";
      return s + "s";
    }

    async function pollStatus() {
      try {
        const resp = await fetch(statusUrl);
        if (!resp.ok) {
          systemStatusEl.textContent = "DISCONNECTED · Control plane unreachable";
          sysLine1El.textContent = "No heartbeat from Percepta core.";
          sysLine2El.textContent = "Check that web_dashboard.py is running and port 5000 is open.";
          return;
        }
        const data = await resp.json();
        const sys = data.system || {};

        if (!data.available) {
          systemStatusEl.textContent = "STANDBY · Awaiting video ingest";
          sysLine1El.textContent = "Ingest: " + (sys.ingest_source || "—") + " · No frames yet.";
          sysLine2El.textContent = "Confirm file path or camera binding.";
        } else {
          const health = sys.pipeline_health || "Nominal";
          const fps = sys.pipeline_fps != null ? sys.pipeline_fps.toFixed(2) : "—";
          systemStatusEl.textContent = health.toUpperCase() + " · " + fps + " FPS · " + (sys.trackers_active ?? 0) + " tracks";
          sysLine1El.textContent =
            "Ingest: " + (sys.ingest_source || "—") +
            " · " + (sys.ingest_resolution || "—") +
            " · nominal " + (sys.source_fps_nominal || 30) + " fps source";
          sysLine2El.textContent =
            "Uptime " + formatUptime(sys.uptime_sec) +
            " · " + (sys.frames_processed || 0) + " frames · " +
            (sys.model_profile || "—") +
            " · predict every " + (sys.prediction_cadence_frames || 15) + " fr · " +
            (sys.pipeline_detail || "") +
            " · scene sync " + formatTime(data.timestamp);
        }

        overallRiskEl.textContent = formatPct(data.overall_risk);
        setRiskLevel(data.risk_level || "safe");
        riskLevelTextEl.textContent = (data.risk_level || "safe").toUpperCase();

        stampedeValueEl.textContent = formatPct(data.stampede_probability);
        if (data.stampede_probability >= 0.75) {
          stampedeSubEl.textContent = "High risk of stampede – emergency call will be prepared.";
        } else if (data.stampede_probability >= 0.4) {
          stampedeSubEl.textContent = "Elevated stampede probability – closely monitor crowd.";
        } else {
          stampedeSubEl.textContent = "Model does not currently predict a stampede.";
        }

        densityRiskEl.textContent = formatPct(data.density_risk);
        velocityRiskEl.textContent = formatPct(data.velocity_risk);

        if (data.should_auto_call && !pendingAutoTrigger && !callInProgress) {
          autoTriggerIndicatorEl.textContent = "Automatic emergency escalation armed.";
          startCountdown("calling emergency services to the location now (auto-detected risk)");
          pendingAutoTrigger = true;
        } else if (!data.should_auto_call && !callInProgress && pendingAutoTrigger) {
          // Situation improved while timer was visible but no call yet
          autoTriggerIndicatorEl.textContent = "";
          clearCountdown(true);
          pendingAutoTrigger = false;
        }
      } catch (e) {
        systemStatusEl.textContent = "NETWORK FAULT · Dashboard API error";
        sysLine1El.textContent = "Could not poll /status.";
        sysLine2El.textContent = String(e && e.message ? e.message : "Retry in a few seconds.");
      }
    }

    function startCountdown(reasonText) {
      clearCountdown(false);
      manualBtn.disabled = true;
      callInProgress = false;

      let remaining = 10;
      emergencyMessageEl.textContent = "The model is " + reasonText + ". If no action is taken, the system will place the call automatically.";
      timerPillEl.style.display = "inline-flex";
      cancelCallBtn.style.display = "inline";
      timerTextEl.textContent = "Calling in " + remaining + "s…";

      countdownInterval = setInterval(async () => {
        remaining -= 1;
        if (remaining > 0) {
          timerTextEl.textContent = "Calling in " + remaining + "s…";
        } else {
          clearInterval(countdownInterval);
          countdownInterval = null;
          await triggerEmergencyCall();
        }
      }, 1000);
    }

    function clearCountdown(resetMessage) {
      if (countdownInterval) {
        clearInterval(countdownInterval);
        countdownInterval = null;
      }
      timerPillEl.style.display = "none";
      cancelCallBtn.style.display = "none";
      manualBtn.disabled = false;
      if (resetMessage) {
        emergencyMessageEl.textContent = "Emergency call cancelled. Monitoring continues.";
      }
    }

    async function triggerEmergencyCall() {
      if (callInProgress) return;
      callInProgress = true;
      manualBtn.disabled = true;
      timerPillEl.style.display = "none";
      cancelCallBtn.style.display = "none";
      emergencyMessageEl.textContent = "Calling emergency services to the location now…";

      try {
        const resp = await fetch(emergencyUrl, { method: "POST" });
        const data = await resp.json().catch(() => ({}));
        emergencyMessageEl.textContent = (data.message || "Emergency services have been notified.") +
          (data.open_stampede_view ? " Open «Stampede zones» to share hotspot map with dispatch." : "");
        if (data.open_stampede_view) {
          videoEl.src = "/video_feed?view=stampede&_=" + Date.now();
          setActiveViewButton("stampede");
        }
      } catch (e) {
        emergencyMessageEl.textContent = "Attempted to call emergency services, but the backend returned an error. Confirm via normal phone line.";
      }
    }

    async function cancelPendingCall() {
      clearCountdown(true);
      pendingAutoTrigger = false;
      try {
        await fetch(cancelUrl, { method: "POST" });
      } catch (e) {
        // Best-effort; UI has already been reset
      }
    }

    manualBtn.addEventListener("click", () => {
      if (callInProgress) return;
      startCountdown("calling emergency services to the location now (manual override)");
    });

    cancelCallBtn.addEventListener("click", () => {
      cancelPendingCall();
    });

    setInterval(pollStatus, 1000);
    pollStatus();
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    """Main dashboard page."""
    return render_template_string(INDEX_HTML)


if __name__ == "__main__":
    # Run the Flask development server
    app.run(host="0.0.0.0", port=5000, debug=False)

