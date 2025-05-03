# app.py
import flask
from flask import Flask, request, jsonify, Response, render_template, send_from_directory
from flask_cors import CORS
import threading
import cv2
import numpy as np
import base64
import json
# Make sure detector.py is in the same directory or Python path
try:
    from detector import OvercrowdingDetector
except ImportError:
    print("ERROR: Cannot import OvercrowdingDetector from detector.py.")
    print("Ensure detector.py exists and is in the same directory or accessible in your PYTHONPATH.")
    # Exit or raise an error if the detector is critical for startup
    # exit(1)
    # For now, let's define a dummy class so the rest of the Flask app can load
    # You MUST replace this with your actual detector.py content
    print("WARNING: Using a dummy OvercrowdingDetector class. Detection will not work.")
    class OvercrowdingDetector:
        def __init__(self, use_camera=False, model_path=None, deep_sort_weights=None):
            self.threshold = 5
            self.fps = 0
            print(f"Dummy Detector Initialized (use_camera={use_camera}, model={model_path}, deep_sort={deep_sort_weights})")
            if not use_camera:
                 # Simulate requiring model files even for dummy
                 if not os.path.exists(model_path or ""):
                      print(f"Warning: Dummy detector checking for model path '{model_path}', not found.")
                 if not os.path.exists(deep_sort_weights or ""):
                     print(f"Warning: Dummy detector checking for deep_sort weights '{deep_sort_weights}', not found.")

        def process_single_frame(self, frame):
            print("Dummy Detector: Processing frame")
            # Simulate processing: just return the frame, 0 count, no alert
            h, w = frame.shape[:2]
            cv2.putText(frame, "Dummy Output", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.fps = 10 # Dummy FPS
            people_count = 0
            alert = people_count > self.threshold
            return frame, people_count, alert

        def cleanup(self):
             print("Dummy Detector: Cleanup called")


import time
import os
import traceback

app = Flask(__name__)
# Allow all origins for development ease. Restrict in production.
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Directories
BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, 'static')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATE_DIR, exist_ok=True)

# --- Model Paths ---
# Define paths relative to the app.py file
MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
DEEP_SORT_WEIGHTS = os.path.join(BASE_DIR, "deep_sort/deep/checkpoint/ckpt.t7")

# Global state
detector_instance = None
# No separate detector_thread needed if processing happens within the request
detection_stats = {
    "people_count": 0,
    "threshold": 5,
    "alert": False,
    "last_updated": None,
    "fps": 0
}
default_threshold = 5

# --- Routes ---

@app.route("/")
def index():
    template_path = os.path.join(TEMPLATE_DIR, 'index.html')
    if not os.path.exists(template_path):
        print("index.html not found, generating...")
        save_html_template()
    return render_template("index.html")

# --- Helper: Check Model Files ---
def check_model_files():
    missing_files = []
    if not os.path.exists(MODEL_PATH):
        missing_files.append(f"YOLO Model: {MODEL_PATH}")
    if not os.path.exists(DEEP_SORT_WEIGHTS):
        # Depending on your detector, DeepSORT might be optional
        print(f"Warning: DeepSORT weights not found at {DEEP_SORT_WEIGHTS}. Tracking might be affected.")
        # missing_files.append(f"DeepSORT Weights: {DEEP_SORT_WEIGHTS}") # Uncomment if mandatory
    return missing_files

# --- Start Detection Route ---
@app.route("/start", methods=["POST", "OPTIONS"])
def start_detection():
    global detector_instance, detection_stats

    if request.method == "OPTIONS":
        response = jsonify({'message': 'preflight'})
        # Be specific with origins in production, e.g., response.headers.add('Access-Control-Allow-Origin', 'https://yourfrontend.com')
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response

    try:
        data = request.get_json(force=True, silent=True) or {}
        threshold = data.get("threshold", default_threshold)

        try:
            threshold = int(threshold)
            if threshold <= 0:
                threshold = default_threshold
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid threshold value. Must be a positive integer."}), 400

        if detector_instance:
            print(f"Detection already running. Updating threshold to {threshold}")
            detector_instance.threshold = threshold
            detection_stats["threshold"] = threshold
            return jsonify({"message": f"Detection running, threshold updated to {threshold}"}), 200

        # Check for required model files before initializing
        missing = check_model_files()
        if missing:
             error_message = "Cannot start detection. Missing required file(s): " + ", ".join(missing)
             print(f"ERROR: {error_message}")
             return jsonify({"error": error_message}), 500

        # Reset stats for a fresh start
        detection_stats = {
            "people_count": 0,
            "threshold": threshold,
            "alert": False,
            "last_updated": time.time(),
            "fps": 0
        }

        try:
            print(f"Initializing detector with threshold: {threshold}")
            # Pass absolute or correct relative paths
            detector_instance = OvercrowdingDetector(
                use_camera=False, # We process frames sent from client
                model_path=MODEL_PATH,
                deep_sort_weights=DEEP_SORT_WEIGHTS
            )
            detector_instance.threshold = threshold
            print("Detector initialized successfully.")
            return jsonify({"message": f"Detection started with threshold {threshold}"}), 200

        except Exception as e:
            print("--- Error during Detector Initialization ---")
            traceback.print_exc()
            print("--- End Error Traceback ---")
            # Provide a generic error message to the client
            return jsonify({"error": f"Failed to initialize the detection engine. Please check server logs."}), 500

    except Exception as e:
        print(f"Error parsing start request: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Error processing start request: {str(e)}"}), 400


# --- Process Frame Route ---
@app.route("/process_frame", methods=["POST"])
def process_frame():
    global detector_instance, detection_stats

    if not detector_instance:
        # print("[ERROR] /process_frame: Detector not initialized") # Less verbose logging
        return jsonify({"error": "Detector not initialized. Please start detection first."}), 400

    if not request.is_json:
         # print(f"[ERROR] /process_frame: Invalid content type: {request.content_type}")
         return jsonify({"error": "Request content type must be application/json"}), 415

    data = request.get_json()
    if not data or "frame" not in data:
         # print("[ERROR] /process_frame: Missing or invalid JSON data")
         return jsonify({"error": "Missing 'frame' key or invalid JSON data"}), 400

    frame_data_uri = data["frame"]
    if not isinstance(frame_data_uri, str) or not frame_data_uri.startswith('data:image/jpeg;base64,'):
        # print(f"[ERROR] /process_frame: Invalid frame data format.")
        return jsonify({"error": "Invalid frame data format (expecting data:image/jpeg;base64,...)"}), 400

    try:
        # Decode
        encoded_frame = frame_data_uri.split(',', 1)[1]
        img_bytes = base64.b64decode(encoded_frame)
        if not img_bytes: raise ValueError("Decoded image bytes are empty")

        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
             raise ValueError("cv2.imdecode failed or returned empty frame")

    except (base64.binascii.Error, ValueError, Exception) as decode_err:
        print(f"[ERROR] /process_frame: Error decoding frame: {decode_err}")
        # traceback.print_exc() # Only enable for deep debugging
        return jsonify({"error": f"Error decoding frame data: {str(decode_err)}"}), 400

    try:
        # Process
        start_time = time.time()
        processed_frame, people_count, alert = detector_instance.process_single_frame(frame)
        processing_time = time.time() - start_time
        current_fps = 1.0 / processing_time if processing_time > 0 else 0

        # Update stats (with smoothing for FPS)
        detection_stats["people_count"] = int(people_count)
        detection_stats["alert"] = bool(alert)
        detection_stats["last_updated"] = time.time()
        detection_stats["fps"] = int(round((detection_stats.get("fps", 0) * 0.9) + (current_fps * 0.1)))
        detection_stats["threshold"] = detector_instance.threshold # Ensure sync

    except Exception as detector_error:
         print(f"[ERROR] /process_frame: Exception during detector processing:")
         traceback.print_exc()
         # Avoid sending potentially sensitive internal error details
         return jsonify({"error": "An error occurred during frame processing on the server."}), 500

    try:
        # Encode response frame
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85] # Quality 85
        success, buffer = cv2.imencode('.jpg', processed_frame, encode_param)
        if not success or buffer is None:
             raise ValueError("cv2.imencode failed to encode processed frame.")

        processed_frame_b64 = base64.b64encode(buffer).decode('utf-8')

        # Send response
        return jsonify({
            "processed_frame": f"data:image/jpeg;base64,{processed_frame_b64}",
            "stats": detection_stats
        })

    except Exception as encode_err:
        print(f"[ERROR] /process_frame: Error encoding response frame: {encode_err}")
        return jsonify({"error": "Error encoding processed frame result."}), 500


# --- Status Route ---
@app.route("/status", methods=["GET", "OPTIONS"])
def status():
    global detection_stats
    if request.method == "OPTIONS":
        response = jsonify({'message': 'preflight'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET, OPTIONS')
        return response

    current_status = "running" if detector_instance else "not running"
    response_stats = detection_stats.copy() # Return a copy

    if current_status == "running":
         # Optional: Check for staleness if needed
         # if response_stats["last_updated"] and (time.time() - response_stats["last_updated"] > 30):
         #     print("Warning: Status is running, but stats seem stale.")
         pass # Keep current stats
    else:
         # Return default/reset stats when not running, keeping threshold
         response_stats = {
             "people_count": 0,
             "threshold": detection_stats.get("threshold", default_threshold),
             "alert": False,
             "last_updated": None,
             "fps": 0
         }

    return jsonify({
        "status": current_status,
        "stats": response_stats
    }), 200

# --- Update Threshold Route ---
@app.route("/update_threshold", methods=["POST", "OPTIONS"])
def update_threshold():
    global detector_instance, detection_stats

    if request.method == "OPTIONS":
        response = jsonify({'message': 'preflight'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response

    if not detector_instance:
        return jsonify({"error": "Detector not running. Cannot update threshold."}), 400

    if not request.is_json:
        return jsonify({"error": "Request content type must be application/json"}), 415

    data = request.get_json()
    if data is None or "threshold" not in data:
        return jsonify({"error": "Invalid JSON or missing 'threshold' key"}), 400

    try:
        threshold = int(data["threshold"])
        if threshold <= 0:
            return jsonify({"error": "Threshold must be a positive integer."}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid threshold value. Must be an integer."}), 400

    detector_instance.threshold = threshold
    detection_stats["threshold"] = threshold
    detection_stats["last_updated"] = time.time() # Mark update time
    print(f"Threshold updated to {threshold}")
    return jsonify({"message": f"Threshold updated to {threshold}", "new_threshold": threshold}), 200

# --- Stop Detection Route ---
@app.route("/stop", methods=["POST", "OPTIONS"])
def stop_detection():
    global detector_instance, detection_stats

    if request.method == "OPTIONS":
        response = jsonify({'message': 'preflight'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response

    if detector_instance:
        print("Stopping detection...")
        try:
            # Add explicit cleanup if your detector class has it
             if hasattr(detector_instance, 'cleanup') and callable(detector_instance.cleanup):
                  detector_instance.cleanup()
        except Exception as cleanup_err:
             print(f"Error during detector cleanup: {cleanup_err}")
             traceback.print_exc()

        detector_instance = None # Release the instance

        # Reset stats, keeping the last threshold setting
        current_threshold = detection_stats.get("threshold", default_threshold)
        detection_stats = {
            "people_count": 0,
            "threshold": current_threshold,
            "alert": False,
            "last_updated": None,
            "fps": 0
        }
        print("Detection stopped.")
        return jsonify({"message": "Detection stopped successfully"}), 200
    else:
        print("Stop request received but no active detection found.")
        # Reset stats just in case client state is inconsistent
        current_threshold = detection_stats.get("threshold", default_threshold)
        detection_stats = { "people_count": 0, "threshold": current_threshold, "alert": False, "last_updated": None, "fps": 0 }
        return jsonify({"message": "Detection was not running"}), 200


# --- Function to save the enhanced HTML template ---

def save_html_template():
    """Save the enhanced HTML code to the templates directory"""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Crowd Detection</title>
    <!-- Google Font -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <!-- Font Awesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css">
    <style>
        :root {
            --primary-color: #3b82f6; /* Blue 500 */
            --primary-dark: #2563eb; /* Blue 600 */
            --secondary-color: #10b981; /* Emerald 500 */
            --secondary-dark: #059669; /* Emerald 600 */
            --danger-color: #ef4444; /* Red 500 */
            --danger-dark: #dc2626; /* Red 600 */
            --warning-color: #f59e0b; /* Amber 500 */
            --warning-dark: #d97706; /* Amber 600 */
            --info-color: #0ea5e9; /* Sky 500 */

            --bg-color: #f1f5f9; /* Slate 100 */
            --card-bg: #ffffff;
            --text-color: #1e293b; /* Slate 800 */
            --text-muted: #64748b; /* Slate 500 */
            --text-light: #f8fafc; /* Slate 50 */
            --border-color: #e2e8f0; /* Slate 200 */
            --input-bg: #f8fafc; /* Slate 50 */
            --dark-bg: #1e293b; /* Slate 800 */

            --border-radius: 0.5rem; /* 8px */
            --card-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            --card-shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
            --transition-speed: 0.2s;
        }

        *, *::before, *::after {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        html {
            scroll-behavior: smooth;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
            display: flex;
            flex-direction: column;
            min-height: 100vh;
            font-size: 16px;
        }

        .container {
            width: 100%;
            max-width: 1400px; /* Wider container for side-by-side */
            margin: 0 auto;
            padding: 1.5rem; /* 24px */
            flex-grow: 1;
        }

        header {
            background: linear-gradient(90deg, var(--primary-dark), var(--primary-color));
            color: var(--text-light);
            padding: 1rem 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.75rem; /* 12px */
        }

        header h1 {
            margin: 0;
            font-weight: 600;
            font-size: 1.5rem; /* 24px */
        }

        .main-layout {
            display: grid;
            grid-template-columns: 1fr; /* Default single column */
            gap: 1.5rem; /* 24px */
        }

        /* Responsive layout: Side-by-side on larger screens */
        @media (min-width: 1024px) {
            .main-layout {
                 /* Video area takes more space */
                 grid-template-columns: minmax(0, 3fr) minmax(0, 1.5fr);
            }
        }

        .card {
            background-color: var(--card-bg);
            border-radius: var(--border-radius);
            box-shadow: var(--card-shadow);
            padding: 1.5rem; /* 24px */
            overflow: hidden; /* Ensure content stays within rounded corners */
            transition: transform var(--transition-speed) ease, box-shadow var(--transition-speed) ease;
        }
        .card:hover {
            transform: translateY(-2px);
            box-shadow: var(--card-shadow-lg);
        }

        .card-header {
            display: flex;
            align-items: center;
            gap: 0.75rem; /* 12px */
            margin-bottom: 1.25rem; /* 20px */
            padding-bottom: 0.75rem; /* 12px */
            border-bottom: 1px solid var(--border-color);
        }
        .card-header i {
             font-size: 1.25rem; /* 20px */
             color: var(--primary-color);
         }
        .card-header h3 {
             font-size: 1.125rem; /* 18px */
             font-weight: 600;
             color: var(--text-color);
             margin: 0;
        }

        /* Video Feeds Section */
        .video-feeds-grid {
            display: grid;
            grid-template-columns: 1fr; /* Mobile first: Stacked */
            gap: 1.5rem; /* 24px */
        }

        @media (min-width: 768px) { /* Side-by-side from medium screens */
            .video-feeds-grid {
                 grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        .feed-container {
            position: relative;
            background-color: var(--dark-bg); /* Black background */
            border-radius: calc(var(--border-radius) / 1.5);
            overflow: hidden;
            aspect-ratio: 4 / 3; /* Maintain aspect ratio */
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
        }

        .feed-container video,
        .feed-container img {
            display: block;
            width: 100%;
            height: 100%;
            object-fit: cover; /* Cover the container */
            border-radius: calc(var(--border-radius) / 1.5);
        }
         #input-feed, #output-feed {
             display: none; /* Hidden initially */
         }

        .feed-placeholder {
            position: absolute;
            inset: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: var(--text-muted);
            text-align: center;
            padding: 1rem;
            opacity: 1;
            transition: opacity var(--transition-speed) ease;
            background-color: var(--dark-bg); /* Ensure bg covers */
             border-radius: calc(var(--border-radius) / 1.5);
        }
        .feed-placeholder.hidden {
            opacity: 0;
            pointer-events: none;
        }
        .feed-placeholder i {
            font-size: 2.5rem; /* 40px */
            margin-bottom: 0.75rem; /* 12px */
            color: var(--primary-color);
        }
        .feed-placeholder span {
            font-size: 0.875rem; /* 14px */
            font-weight: 500;
        }

        .feed-title {
            text-align: center;
            font-weight: 600;
            color: var(--text-muted);
            margin-top: 0.75rem; /* 12px */
            font-size: 0.875rem; /* 14px */
        }

        /* Loading Overlay */
        .loading-overlay {
            position: absolute;
            inset: 0;
            background-color: rgba(30, 41, 59, 0.85); /* Slate 800 with opacity */
            color: var(--text-light);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            border-radius: calc(var(--border-radius) / 1.5);
            opacity: 0;
            visibility: hidden;
            transition: opacity var(--transition-speed) ease, visibility var(--transition-speed) ease;
            z-index: 10;
            pointer-events: none;
        }
        .loading-overlay.active {
            opacity: 1;
            visibility: visible;
            pointer-events: auto;
        }

        /* Modern CSS Spinner */
        .spinner {
            width: 44px;
            height: 44px;
            animation: spinner-y0fdc1 2s infinite ease;
            transform-style: preserve-3d;
            margin-bottom: 1rem; /* 16px */
        }
        .spinner > div {
            background-color: rgba(59, 130, 246, 0.2); /* primary-color with alpha */
            height: 100%;
            position: absolute;
            width: 100%;
            border: 2px solid var(--primary-color);
            border-radius: 4px;
        }
        .spinner div:nth-of-type(1) { transform: translateZ(-22px) rotateY(180deg); }
        .spinner div:nth-of-type(2) { transform: rotateY(-270deg) translateX(50%); transform-origin: top right; }
        .spinner div:nth-of-type(3) { transform: rotateY(270deg) translateX(-50%); transform-origin: center left; }
        .spinner div:nth-of-type(4) { transform: rotateX(90deg) translateY(-50%); transform-origin: top center; }
        .spinner div:nth-of-type(5) { transform: rotateX(-90deg) translateY(50%); transform-origin: bottom center; }
        .spinner div:nth-of-type(6) { transform: translateZ(22px); }
        @keyframes spinner-y0fdc1 {
            0% { transform: rotate(45deg) rotateX(-25deg) rotateY(25deg); }
            50% { transform: rotate(45deg) rotateX(-385deg) rotateY(25deg); }
            100% { transform: rotate(45deg) rotateX(-385deg) rotateY(385deg); }
        }
        #overlay-message { font-weight: 500; }

        /* Controls Section */
        .controls-grid {
             display: grid;
             gap: 1.25rem; /* 20px */
         }
        .control-group {
            margin-bottom: 0.5rem; /* 8px */
        }
        .control-group label {
            display: block;
            margin-bottom: 0.5rem; /* 8px */
            font-weight: 500;
            font-size: 0.875rem; /* 14px */
            color: var(--text-muted);
        }
        .control-group input, .control-group select {
            width: 100%;
            padding: 0.625rem 0.75rem; /* 10px 12px */
            border: 1px solid var(--border-color);
            border-radius: calc(var(--border-radius) / 1.5);
            background-color: var(--input-bg);
            font-size: 0.9375rem; /* 15px */
            transition: border-color var(--transition-speed) ease, box-shadow var(--transition-speed) ease;
        }
        .control-group input:focus, .control-group select:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2); /* primary-color with alpha */
        }

        .button-group {
            display: flex;
            gap: 0.75rem; /* 12px */
            flex-wrap: wrap;
            margin-top: 0.5rem; /* Add space above buttons */
        }

        button {
            background-color: var(--primary-color);
            color: var(--text-light);
            border: none;
            padding: 0.625rem 1.125rem; /* 10px 18px */
            border-radius: calc(var(--border-radius) / 1.5);
            cursor: pointer;
            font-size: 0.9375rem; /* 15px */
            font-weight: 500;
            transition: background-color var(--transition-speed) ease, transform var(--transition-speed) ease, box-shadow var(--transition-speed) ease;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem; /* 8px */
            white-space: nowrap;
             box-shadow: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        }
        button:hover:not(:disabled) {
            background-color: var(--primary-dark);
            transform: translateY(-1px);
             box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        }
        button:active:not(:disabled) {
            transform: translateY(0);
            box-shadow: inset 0 2px 4px 0 rgb(0 0 0 / 0.05);
        }
        button:disabled {
            background-color: #94a3b8; /* Slate 400 */
            color: #e2e8f0; /* Slate 200 */
            cursor: not-allowed;
            opacity: 0.8;
        }

        #start-btn { background-color: var(--secondary-color); }
        #start-btn:hover:not(:disabled) { background-color: var(--secondary-dark); }
        #stop-btn { background-color: var(--danger-color); }
        #stop-btn:hover:not(:disabled) { background-color: var(--danger-dark); }
        #update-threshold-btn { background-color: var(--info-color); }
        #update-threshold-btn:hover:not(:disabled) { background-color: #0ea5e9; } /* Darker Sky */
        #test-camera-btn { background-color: var(--warning-color); font-size: 0.875rem; padding: 0.5rem 0.75rem; }
        #test-camera-btn:hover:not(:disabled) { background-color: var(--warning-dark); }

        .file-input-wrapper { margin-top: 0.75rem; display: flex; align-items: center; gap: 0.5rem;}
        .file-input-wrapper input[type="file"] { display: none; }
        .file-input-wrapper label { /* Custom button style */
            background-color: var(--text-muted);
            color: var(--text-light);
            padding: 0.5rem 1rem;
            border-radius: calc(var(--border-radius) / 1.5);
            cursor: pointer;
            transition: background-color var(--transition-speed);
            font-size: 0.875rem;
            display: inline-flex; align-items: center; gap: 0.5rem;
        }
        .file-input-wrapper label:hover { background-color: var(--text-color); }
        #file-name { font-size: 0.8125rem; /* 13px */ color: var(--text-muted); font-style: italic; }

        /* Stats Section */
        .stats-list {
             list-style: none;
             padding: 0;
             margin: 0;
             display: grid;
             gap: 0.875rem; /* 14px */
         }
        .stats-item {
             display: flex;
             justify-content: space-between;
             align-items: center;
             padding: 0.625rem 0; /* 10px 0 */
             border-bottom: 1px solid var(--border-color);
             font-size: 0.9375rem; /* 15px */
         }
         .stats-item:last-child { border-bottom: none; }
         .stats-label {
             color: var(--text-muted);
             font-weight: 500;
             display: flex;
             align-items: center;
             gap: 0.5rem;
         }
         .stats-value {
             font-weight: 600;
             color: var(--text-color);
             display: flex;
             align-items: center;
             gap: 0.5rem; /* Space for indicator */
         }

        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            transition: background-color var(--transition-speed) ease;
            flex-shrink: 0; /* Prevent shrinking */
        }
        .status-running {
            background-color: var(--secondary-color);
            animation: pulse 1.5s infinite;
        }
        .status-stopped {
            background-color: var(--danger-color);
        }
        .status-alert-active {
            background-color: var(--warning-color);
             /* Optional: Add subtle animation for active alert */
             animation: pulse-alert 1s infinite alternate;
        }
         .status-alert-inactive {
            background-color: #94a3b8; /* Slate 400 */
         }

        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.6); } /* secondary-color with alpha */
            70% { box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
            100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }
        @keyframes pulse-alert {
             from { opacity: 1; }
             to { opacity: 0.6; }
         }

        /* Alert Banner */
        .alert-banner {
            background-color: var(--danger-color);
            color: var(--text-light);
            padding: 0.875rem 1.25rem; /* 14px 20px */
            border-radius: var(--border-radius);
            margin-bottom: 1.5rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            opacity: 0;
            transform: translateY(-10px) scale(0.98);
            transition: opacity var(--transition-speed) ease, transform var(--transition-speed) ease;
            visibility: hidden;
        }
        .alert-banner.visible {
            opacity: 1;
            transform: translateY(0) scale(1);
            visibility: visible;
        }
        .alert-banner i { font-size: 1.25rem; }
        .alert-flash { animation: flash 1.2s infinite ease-in-out; }
        @keyframes flash {
            0%, 100% { background-color: var(--danger-color); }
            50% { background-color: var(--danger-dark); }
        }

        /* Debug Panel */
         #debug-panel {
            position: fixed;
            bottom: 10px;
            right: 10px;
            background: rgba(30, 41, 59, 0.9); /* Slate 800 */
            color: var(--text-light);
            padding: 1rem;
            font-family: monospace;
            font-size: 0.75rem; /* 12px */
            z-index: 9999;
            max-width: 90vw;
             width: 500px;
            max-height: 300px;
            overflow: auto;
            border-radius: var(--border-radius);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
             border: 1px solid var(--text-muted);
             opacity: 0; visibility: hidden;
             transform: translateY(10px);
             transition: opacity var(--transition-speed) ease, visibility var(--transition-speed) ease, transform var(--transition-speed) ease;
        }
        #debug-panel.visible {
            opacity: 1; visibility: visible; transform: translateY(0);
        }
        #debug-panel-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--text-muted); padding-bottom: 0.5rem; margin-bottom: 0.75rem; }
        #debug-panel-header span { font-weight: bold; color: var(--info-color); }
        #debug-panel-header button { background: none; border: none; color: var(--text-muted); font-size: 1rem; cursor: pointer; padding: 0.25rem; line-height: 1;}
        #debug-panel-header button:hover { color: var(--text-light); }
        #debug-info { white-space: pre-wrap; word-break: break-all; }
        #debug-panel-controls { margin-top: 0.75rem; display: flex; gap: 0.5rem;}
        #debug-panel-controls button { font-size: 0.75rem; background-color: var(--text-muted); padding: 0.3rem 0.6rem; }
        #debug-panel-controls button:hover { background-color: #475569; } /* Slate 600 */

         #debug-panel-toggle {
             position: fixed;
             bottom: 1rem; /* 16px */
             left: 1rem; /* 16px */
             background: var(--warning-color);
             color: var(--text-light);
             border: none;
             width: 44px;
             height: 44px;
             border-radius: 50%;
             cursor: pointer;
             z-index: 9998;
             font-size: 1.25rem; /* 20px */
             box-shadow: 0 2px 5px rgba(0,0,0,0.2);
             display: flex; align-items: center; justify-content: center;
             transition: background-color var(--transition-speed) ease, transform var(--transition-speed) ease;
         }
         #debug-panel-toggle:hover { background-color: var(--warning-dark); transform: scale(1.1); }
         #debug-panel-toggle.active { background-color: var(--danger-color); } /* Indicate when panel is open */

         footer {
             text-align: center;
             padding: 1.5rem;
             margin-top: 2rem;
             background-color: var(--card-bg);
             color: var(--text-muted);
             font-size: 0.875rem; /* 14px */
             border-top: 1px solid var(--border-color);
         }
         footer a {
             color: var(--primary-color);
             text-decoration: none;
             font-weight: 500;
         }
         footer a:hover {
             text-decoration: underline;
             color: var(--primary-dark);
         }

    </style>
</head>
<body>
    <!-- Debug Panel Toggle Button -->
    <button id="debug-panel-toggle" title="Toggle Debug Panel">
        <i class="fa-solid fa-bug"></i>
    </button>

    <!-- Debug Panel -->
    <div id="debug-panel">
        <div id="debug-panel-header">
            <span><i class="fa-solid fa-terminal"></i> Debug Console</span>
            <button onclick="toggleDebugPanel(false)" title="Close Panel">×</button>
        </div>
        <div id="debug-info">Initializing debug log...</div>
        <div id="debug-panel-controls">
            <button onclick="clearDebugInfo()"><i class="fa-solid fa-eraser"></i> Clear</button>
        </div>
    </div>

    <header>
        <i class="fa-solid fa-video"></i>
        <h1>Advanced Crowd Detection</h1>
    </header>

    <div class="container">
        <!-- Alert Banner -->
        <div class="alert-banner" id="alert-banner">
            <i class="fa-solid fa-triangle-exclamation"></i>
            <span id="alert-message">ALERT: Overcrowding threshold exceeded!</span>
        </div>

        <div class="main-layout">

            <!-- Video Feeds Section -->
            <section class="video-feeds-section card">
                 <div class="card-header">
                     <i class="fa-solid fa-camera-retro"></i>
                     <h3>Live Video Feeds</h3>
                 </div>
                 <div class="video-feeds-grid">
                     <!-- Input Feed Column -->
                     <div>
                         <div class="feed-container">
                             <video id="input-feed" autoplay muted playsinline></video>
                             <div class="feed-placeholder" id="input-placeholder">
                                 <i class="fa-solid fa-video-slash"></i>
                                 <span>Webcam Feed Inactive</span>
                             </div>
                             <div class="loading-overlay" id="input-overlay">
                                  <div class="spinner"><div></div><div></div><div></div><div></div><div></div><div></div></div>
                                 <span id="input-overlay-message">Waiting...</span>
                             </div>
                         </div>
                         <p class="feed-title">Input Feed (Raw)</p>
                     </div>

                     <!-- Output Feed Column -->
                     <div>
                         <div class="feed-container">
                             <img id="output-feed" alt="Processed Video Feed">
                             <div class="feed-placeholder" id="output-placeholder">
                                 <i class="fa-solid fa-microchip"></i>
                                 <span>Processed Feed Inactive</span>
                             </div>
                             <div class="loading-overlay" id="output-overlay">
                                  <div class="spinner"><div></div><div></div><div></div><div></div><div></div><div></div></div>
                                 <span id="output-overlay-message">Waiting for processing...</span>
                             </div>
                         </div>
                         <p class="feed-title">Output Feed (Processed)</p>
                     </div>
                 </div>
            </section>

            <!-- Controls and Stats Section -->
            <section class="controls-stats-section">
                <div class="card">
                     <div class="card-header">
                         <i class="fa-solid fa-sliders"></i>
                         <h3>Detection Controls</h3>
                     </div>
                     <div class="controls-grid">
                         <div class="control-group">
                            <label for="video-source"><i class="fa-solid fa-video"></i> Select Video Source</label>
                            <select id="video-source">
                                <option value="webcam" selected>Webcam</option>
                                <!-- <option value="file">Upload Video File</option> -->
                            </select>
                        </div>

                        <div id="webcam-controls">
                             <div class="control-group">
                                <label for="webcam-selector"><i class="fa-solid fa-camera"></i> Select Camera</label>
                                <select id="webcam-selector">
                                    <option value="">Loading cameras...</option>
                                </select>
                            </div>
                             <button id="test-camera-btn">
                                 <i class="fa-solid fa-vial-video"></i> Test Camera
                             </button>
                        </div>

                        <div id="file-controls" style="display: none;">
                             <div class="control-group">
                                 <label><i class="fa-solid fa-file-video"></i> Upload Video</label>
                                 <div class="file-input-wrapper">
                                     <input type="file" id="video-file" accept="video/*">
                                     <label for="video-file"><i class="fa-solid fa-upload"></i> Choose File</label>
                                     <span id="file-name">No file selected</span>
                                 </div>
                             </div>
                        </div>

                        <div class="control-group">
                             <label for="threshold-input"><i class="fa-solid fa-users-viewfinder"></i> Overcrowding Threshold</label>
                             <input type="number" id="threshold-input" min="1" value="5">
                         </div>

                        <div class="button-group">
                            <button id="start-btn"><i class="fa-solid fa-play"></i> Start Detection</button>
                            <button id="stop-btn" disabled><i class="fa-solid fa-stop"></i> Stop Detection</button>
                            <button id="update-threshold-btn" disabled><i class="fa-solid fa-arrows-rotate"></i> Update Threshold</button>
                        </div>
                    </div>
                </div>

                <div class="card">
                     <div class="card-header">
                         <i class="fa-solid fa-chart-line"></i>
                         <h3>Detection Statistics</h3>
                     </div>
                     <ul class="stats-list">
                        <li class="stats-item">
                            <span class="stats-label"><i class="fa-solid fa-power-off"></i> Status</span>
                            <span class="stats-value">
                                <span class="status-indicator status-stopped" id="status-indicator"></span>
                                <span id="status-text">Not Running</span>
                            </span>
                        </li>
                        <li class="stats-item">
                            <span class="stats-label"><i class="fa-solid fa-users"></i> People Count</span>
                            <span class="stats-value" id="people-count">0</span>
                        </li>
                        <li class="stats-item">
                            <span class="stats-label"><i class="fa-solid fa-bullseye"></i> Threshold</span>
                            <span class="stats-value" id="threshold-display">5</span>
                        </li>
                        <li class="stats-item">
                            <span class="stats-label"><i class="fa-solid fa-bell"></i> Alert Status</span>
                            <span class="stats-value">
                                 <span class="status-indicator status-alert-inactive" id="alert-indicator"></span>
                                <span id="alert-status">Inactive</span>
                            </span>
                        </li>
                        <li class="stats-item">
                            <span class="stats-label"><i class="fa-solid fa-gauge-high"></i> Processing FPS</span>
                            <span class="stats-value" id="fps-display">0</span>
                        </li>
                        <li class="stats-item">
                            <span class="stats-label"><i class="fa-regular fa-clock"></i> Last Update</span>
                            <span class="stats-value" id="last-updated">Never</span>
                        </li>
                    </ul>
                </div>
            </section>

        </div> <!-- End Main Layout -->
    </div> <!-- End Container -->

    <footer>
        Advanced Crowd Detection System © 2024 | Built with Flask & OpenCV | <a href="https://github.com" target="_blank" rel="noopener noreferrer">View on GitHub (Example Link)</a>
    </footer>

    <script>
        // === DOM Elements ===
        const startBtn = document.getElementById('start-btn');
        const stopBtn = document.getElementById('stop-btn');
        const thresholdInput = document.getElementById('threshold-input');
        const updateThresholdBtn = document.getElementById('update-threshold-btn');

        // Video Feed Elements
        const inputFeed = document.getElementById('input-feed');
        const outputFeed = document.getElementById('output-feed');
        const inputPlaceholder = document.getElementById('input-placeholder');
        const outputPlaceholder = document.getElementById('output-placeholder');
        const inputOverlay = document.getElementById('input-overlay');
        const outputOverlay = document.getElementById('output-overlay');
        const inputOverlayMsg = document.getElementById('input-overlay-message');
        const outputOverlayMsg = document.getElementById('output-overlay-message');

        // Status & Stats Elements
        const statusIndicator = document.getElementById('status-indicator');
        const statusText = document.getElementById('status-text');
        const peopleCount = document.getElementById('people-count');
        const thresholdDisplay = document.getElementById('threshold-display');
        const alertIndicator = document.getElementById('alert-indicator');
        const alertStatus = document.getElementById('alert-status');
        const fpsDisplay = document.getElementById('fps-display');
        const lastUpdated = document.getElementById('last-updated');
        const alertBanner = document.getElementById('alert-banner');
        const alertMessage = document.getElementById('alert-message');

        // Control Elements
        const videoSourceSelect = document.getElementById('video-source');
        const webcamControls = document.getElementById('webcam-controls');
        const fileControls = document.getElementById('file-controls');
        const webcamSelector = document.getElementById('webcam-selector');
        const testCameraBtn = document.getElementById('test-camera-btn');
        const videoFileInput = document.getElementById('video-file');
        const fileNameDisplay = document.getElementById('file-name');

        // Debug Panel Elements
        const debugPanel = document.getElementById('debug-panel');
        const debugInfo = document.getElementById('debug-info');
        const debugToggleBtn = document.getElementById('debug-panel-toggle');

        // === API Endpoints ===
        const API_BASE_URL = ''; // Keep empty if Flask serves from same origin
        const STATUS_ENDPOINT = `${API_BASE_URL}/status`;
        const START_ENDPOINT = `${API_BASE_URL}/start`;
        const PROCESS_FRAME_ENDPOINT = `${API_BASE_URL}/process_frame`;
        const UPDATE_THRESHOLD_ENDPOINT = `${API_BASE_URL}/update_threshold`;
        const STOP_ENDPOINT = `${API_BASE_URL}/stop`;

        // === App State ===
        let isDetectionRunning = false;
        let frameProcessingTimeout = null;
        let selectedCameraId = null;
        let mediaStream = null; // For the raw webcam input feed
        let consecutiveErrors = 0;
        const frameInterval = 150; // ms between successful frame sends (adjust for performance)
        const retryDelayBase = 500; // ms initial delay on error
        const maxConsecutiveErrors = 8; // Stop after this many errors
        const requestTimeout = 8000; // ms for fetch requests (increased slightly)

        let debugLog = [];
        let isDebugVisible = false;
        const MAX_DEBUG_LINES = 75;

        // === Logging & Debugging ===
        const originalLog = console.log;
        const originalError = console.error;
        const originalWarn = console.warn;

        function logToDebug(type, args) {
            try {
                const message = Array.from(args).map(arg =>
                    typeof arg === 'object' ? JSON.stringify(arg, null, 2) : String(arg)
                ).join(' ');
                const timestamp = new Date().toLocaleTimeString();
                const colorMap = { log: '#90ee90', error: '#ff6b6b', warn: '#ffa500', info: '#87ceeb' };
                const logEntry = `<span style="color: ${colorMap[type] || '#ffffff'};">[${timestamp} ${type.toUpperCase()}]</span> ${message.replace(/</g, "<").replace(/>/g, ">")}`; // Basic escaping

                debugLog.push(logEntry);
                if (debugLog.length > MAX_DEBUG_LINES) debugLog.shift();

                if (debugInfo && isDebugVisible) {
                    debugInfo.innerHTML = debugLog.join('<br>');
                    debugInfo.scrollTop = debugInfo.scrollHeight;
                }
            } catch (e) {
                 originalError("Error in logToDebug:", e); // Avoid log loops
            }
        }

        console.log = function() { originalLog.apply(console, arguments); logToDebug('log', arguments); };
        console.error = function() { originalError.apply(console, arguments); logToDebug('error', arguments); };
        console.warn = function() { originalWarn.apply(console, arguments); logToDebug('warn', arguments); };
        console.info = function() { originalLog.apply(console, arguments); logToDebug('info', arguments); }; // Add info level

        function clearDebugInfo() {
            debugLog = [];
            if (debugInfo) debugInfo.innerHTML = '';
            console.info("Debug log cleared.");
        }

        function toggleDebugPanel(show) {
             isDebugVisible = (show === undefined) ? !isDebugVisible : show;
             if (debugPanel) {
                 debugPanel.classList.toggle('visible', isDebugVisible);
                 if(isDebugVisible) {
                     debugInfo.innerHTML = debugLog.join('<br>');
                     debugInfo.scrollTop = debugInfo.scrollHeight;
                 }
             }
             debugToggleBtn.classList.toggle('active', isDebugVisible);
             debugToggleBtn.innerHTML = isDebugVisible ? '<i class="fa-solid fa-times"></i>' : '<i class="fa-solid fa-bug"></i>';
             debugToggleBtn.title = isDebugVisible ? 'Hide Debug Panel' : 'Show Debug Panel';
         }

        // === Initialization ===
        document.addEventListener('DOMContentLoaded', () => {
            console.info("DOM Content Loaded. Initializing UI.");
            setupEventListeners();
            checkDetectionStatus(); // Check server status first
            loadCameras();
            updateUIForStoppedState(); // Set initial UI state
            toggleDebugPanel(false); // Start hidden
            debugInfo.innerHTML = 'Debug panel active. Waiting for logs...';
        });

        // === Event Listeners ===
        function setupEventListeners() {
            startBtn.addEventListener('click', handleStartClick);
            stopBtn.addEventListener('click', handleStopClick);
            updateThresholdBtn.addEventListener('click', handleUpdateThresholdClick);
            videoSourceSelect.addEventListener('change', handleVideoSourceChange);
            webcamSelector.addEventListener('change', handleCameraSelection);
            testCameraBtn.addEventListener('click', testCameraAccess);
            videoFileInput.addEventListener('change', handleFileSelection);
            debugToggleBtn.addEventListener('click', () => toggleDebugPanel());
            thresholdInput.addEventListener('keypress', (e) => {
                 if (e.key === 'Enter' && !updateThresholdBtn.disabled) {
                     handleUpdateThresholdClick();
                 }
            });
            // Add listeners to hide overlays if clicked (optional)
             // inputOverlay.addEventListener('click', () => toggleOverlay('input', false));
             // outputOverlay.addEventListener('click', () => toggleOverlay('output', false));
        }

        // === UI Update Functions ===

        function updateUIForStoppedState(stats) {
            console.info("Updating UI to: Stopped State");
            isDetectionRunning = false;

            // Buttons & Controls
            startBtn.disabled = false;
            stopBtn.disabled = true;
            updateThresholdBtn.disabled = true;
            videoSourceSelect.disabled = false;
            webcamSelector.disabled = false;
            thresholdInput.disabled = false; // Allow editing threshold when stopped

            // Video Feeds
            toggleOverlay('input', false);
            toggleOverlay('output', false);
            inputFeed.style.display = 'none';
            inputFeed.srcObject = null; // Ensure stream is released
            outputFeed.style.display = 'none';
            outputFeed.src = ''; // Clear image
            inputPlaceholder.classList.remove('hidden');
            outputPlaceholder.classList.remove('hidden');

            // Status & Stats
            const defaultStats = { people_count: 0, threshold: parseInt(thresholdInput.value) || 5, alert: false, fps: 0, last_updated: null };
            updateStatsDisplay(stats || defaultStats); // Update with provided or default stats
            statusIndicator.className = 'status-indicator status-stopped';
            statusText.textContent = 'Not Running';
            alertBanner.classList.remove('visible', 'alert-flash'); // Hide alert banner
        }

        function updateUIForRunningState(initialStats) {
            console.info("Updating UI to: Running State");
            isDetectionRunning = true;

             // Buttons & Controls
            startBtn.disabled = true;
            stopBtn.disabled = false;
            updateThresholdBtn.disabled = false;
            videoSourceSelect.disabled = true; // Lock source while running
            webcamSelector.disabled = true; // Lock camera while running
            thresholdInput.disabled = true; // Lock threshold input field while running (use button)

            // Video Feeds (Placeholders hidden by stream start logic)
            toggleOverlay('input', false); // Ensure overlays are initially hidden
            toggleOverlay('output', false);
            outputFeed.style.display = 'block'; // Prepare output area
            outputPlaceholder.classList.add('hidden'); // Hide output placeholder

            // Status & Stats
             updateStatsDisplay(initialStats || { threshold: parseInt(thresholdInput.value) || 5 }); // Update with initial/current threshold
            statusIndicator.className = 'status-indicator status-running';
            statusText.textContent = 'Running';
        }

         function updateStatsDisplay(stats) {
            if (!stats) { console.warn("updateStatsDisplay called with null stats."); return; }

            peopleCount.textContent = stats.people_count ?? 'N/A';
            thresholdDisplay.textContent = stats.threshold ?? 'N/A';
             // Keep threshold input value synced if needed, though locked when running
             // thresholdInput.value = stats.threshold ?? 5;
            fpsDisplay.textContent = stats.fps ?? 'N/A';

            const alertActive = stats.alert === true;
            alertIndicator.className = `status-indicator ${alertActive ? 'status-alert-active' : 'status-alert-inactive'}`;
            alertStatus.textContent = alertActive ? 'Active' : 'Inactive';
            alertBanner.classList.toggle('visible', alertActive);
            alertBanner.classList.toggle('alert-flash', alertActive);
             if(alertActive) {
                 alertMessage.textContent = `ALERT: People count (${stats.people_count}) exceeds threshold (${stats.threshold})!`;
             }

            if (stats.last_updated) {
                try {
                    const date = new Date(stats.last_updated * 1000);
                    lastUpdated.textContent = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
                } catch (e) { lastUpdated.textContent = "Invalid Date"; }
            } else {
                lastUpdated.textContent = isDetectionRunning ? "Waiting..." : "Never";
            }
        }

        function toggleOverlay(feedType, show, message = "Loading...") {
            const overlay = feedType === 'input' ? inputOverlay : outputOverlay;
            const msgElement = feedType === 'input' ? inputOverlayMsg : outputOverlayMsg;
            if (overlay) {
                msgElement.textContent = message;
                overlay.classList.toggle('active', show);
                // console.log(`Overlay ${feedType} toggled: ${show ? 'ON' : 'OFF'} - ${message}`);
            }
        }

        // === Core Logic Handlers ===

        async function handleStartClick() {
            if (isDetectionRunning) return;

            const source = videoSourceSelect.value;
            if (source === 'webcam' && !selectedCameraId) {
                alert("Please select a camera from the dropdown first.");
                console.error("Start aborted: No webcam selected.");
                return;
            }
             if (source === 'file' && !videoFileInput.files[0]) {
                 alert("Please choose a video file to upload first.");
                 console.error("Start aborted: No file selected.");
                 return;
             }

            console.info("Attempting to start detection...");
            startBtn.disabled = true; // Disable immediately
            toggleOverlay('input', true, 'Initializing...');
            toggleOverlay('output', true, 'Initializing...');

            try {
                const threshold = parseInt(thresholdInput.value);
                if (isNaN(threshold) || threshold <= 0) {
                    throw new Error("Threshold must be a positive number.");
                }

                console.log(`Sending start request to server (threshold: ${threshold})...`);
                const response = await fetch(START_ENDPOINT, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ threshold })
                });
                const data = await response.json();

                if (!response.ok) throw new Error(data.error || `Server error: ${response.status}`);

                console.info("Server started detection:", data.message);
                updateUIForRunningState({ threshold: threshold }); // Set UI to running state

                // Now start the client-side stream/processing
                if (source === 'webcam') {
                    await startWebcamStreamAndProcessing(); // This handles its own overlays
                } else {
                     console.warn("File processing not yet implemented.");
                     alert("Video file processing is not available in this version.");
                     await handleStopClick(); // Stop if file processing isn't ready
                     toggleOverlay('input', false); toggleOverlay('output', false); // Hide overlays if stopped
                }
                 // Overlays are managed by startWebcamStreamAndProcessing or error handling

            } catch (error) {
                console.error('Error starting detection:', error);
                alert(`Failed to start detection: ${error.message}`);
                 updateUIForStoppedState(); // Revert UI
                 toggleOverlay('input', false); toggleOverlay('output', false); // Hide overlays on failure
                startBtn.disabled = false; // Re-enable button
            }
        }

         async function handleStopClick() {
             // Allow stopping even if client state thinks it's not running, to ensure server stops
             console.info("Attempting to stop detection...");
             stopBtn.disabled = true; // Disable immediately
             toggleOverlay('input', true, 'Stopping...');
             toggleOverlay('output', true, 'Stopping...');

             // 1. Stop client-side processing & streams first
             stopFrameProcessingLoop();
             stopWebcamStream(); // Stops the input feed

             // 2. Send stop request to server
             try {
                 console.log("Sending stop request to server...");
                 const response = await fetch(STOP_ENDPOINT, { method: 'POST' });
                 const data = await response.json();
                 if (!response.ok) {
                     console.warn(`Server reported issue during stop: ${data.message || response.statusText}`);
                     // Continue cleanup anyway
                 } else {
                     console.info("Server confirmed stop:", data.message);
                 }
             } catch (error) {
                 console.error('Network error sending stop request:', error);
                 alert(`Network error stopping detection: ${error.message}. UI may be out of sync.`);
             } finally {
                 // 3. Update UI to stopped state definitively
                 updateUIForStoppedState(); // This handles stats reset and overlay hiding
                 console.info("Client-side detection stopped.");
             }
         }

        async function handleUpdateThresholdClick() {
            if (!isDetectionRunning) {
                alert("Detection is not running. Start detection to update the threshold.");
                return;
            }

            const threshold = parseInt(thresholdInput.value);
            if (isNaN(threshold) || threshold <= 0) {
                alert("Please enter a valid positive number for the threshold.");
                return;
            }

            console.info(`Updating threshold to: ${threshold}`);
            updateThresholdBtn.disabled = true; // Disable during request

            try {
                 const response = await fetch(UPDATE_THRESHOLD_ENDPOINT, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ threshold })
                });
                const data = await response.json();
                 if (!response.ok) throw new Error(data.error || `Server error: ${response.status}`);

                console.info("Server confirmed threshold update:", data.message);
                // Update display based on server response
                const newThreshold = data.new_threshold !== undefined ? data.new_threshold : threshold;
                thresholdDisplay.textContent = newThreshold;
                 thresholdInput.value = newThreshold; // Sync input field as well
                 alert(`Threshold updated to ${newThreshold} successfully!`);

            } catch (error) {
                console.error('Error updating threshold:', error);
                alert(`Failed to update threshold: ${error.message}`);
                // Optionally revert display if needed, or leave as is
            } finally {
                // Re-enable only if still running
                 if (isDetectionRunning) updateThresholdBtn.disabled = false;
            }
        }

        // === Webcam and Frame Processing ===

        async function loadCameras() {
            console.info("Loading cameras...");
            webcamSelector.innerHTML = '<option value="">Detecting...</option>';
            webcamSelector.disabled = true;

            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                console.error("getUserMedia not supported.");
                webcamSelector.innerHTML = '<option value="">Not Supported</option>';
                alert("Camera access (getUserMedia) is not supported by your browser.");
                return;
            }

            try {
                // Request permission first
                console.log("Requesting camera permissions...");
                const tempStream = await navigator.mediaDevices.getUserMedia({ video: true });
                console.log("Permission granted.");
                tempStream.getTracks().forEach(track => track.stop()); // Stop temp stream

                // Enumerate devices
                console.log("Enumerating video devices...");
                const devices = await navigator.mediaDevices.enumerateDevices();
                const videoDevices = devices.filter(device => device.kind === 'videoinput');
                console.log(`Found ${videoDevices.length} video devices.`);

                webcamSelector.innerHTML = ''; // Clear "Detecting..."
                if (videoDevices.length === 0) {
                    webcamSelector.innerHTML = '<option value="">No cameras found</option>';
                    return;
                }

                const defaultOption = document.createElement('option');
                defaultOption.value = ""; defaultOption.text = "-- Select a Camera --";
                webcamSelector.appendChild(defaultOption);

                videoDevices.forEach((device, index) => {
                    const option = document.createElement('option');
                    option.value = device.deviceId;
                    option.text = device.label || `Camera ${index + 1}`;
                    webcamSelector.appendChild(option);
                });
                webcamSelector.disabled = false;
                console.info("Camera list populated.");

            } catch (error) {
                console.error('Error loading cameras:', error.name, error.message);
                webcamSelector.innerHTML = '<option value="">Access Failed</option>';
                 let errorMsg = `Failed to access camera: ${error.name}. `;
                 if (error.name === 'NotAllowedError') errorMsg += "Permission denied.";
                 else if (error.name === 'NotFoundError') errorMsg += "No camera found.";
                 else errorMsg += "See console for details.";
                 alert(errorMsg);
            }
        }

        function handleCameraSelection() {
            selectedCameraId = webcamSelector.value;
            console.log(`Camera selected: ID=${selectedCameraId}`);
            stopWebcamStream(); // Stop any previous stream/preview

            if (selectedCameraId) {
                 // Start preview immediately if a camera is selected, even if not running detection
                 startWebcamPreviewOnly();
             } else {
                  // No camera selected, ensure input feed is cleared
                  inputFeed.style.display = 'none';
                  inputPlaceholder.classList.remove('hidden');
             }
             // If detection was running, user needs to stop/start to use new camera
             if (isDetectionRunning) {
                 alert("Detection is running. Please stop and restart detection to use the newly selected camera.");
                 // Or automatically restart:
                 // handleStopClick().then(() => handleStartClick());
             }
        }

         async function startWebcamPreviewOnly() {
             if (!selectedCameraId) return;
             console.log("Starting webcam preview...");
             toggleOverlay('input', true, 'Starting preview...');

             const constraints = { video: { deviceId: { exact: selectedCameraId } } };
             try {
                 mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
                 inputFeed.srcObject = mediaStream;
                 inputFeed.style.display = 'block';
                 inputPlaceholder.classList.add('hidden');
                 toggleOverlay('input', false); // Hide overlay on success
                 console.info("Webcam preview started.");
             } catch (error) {
                 console.error(`Error starting preview for camera ${selectedCameraId}:`, error);
                 alert(`Failed to start preview for camera: ${error.name}`);
                 toggleOverlay('input', false);
                 inputFeed.style.display = 'none';
                 inputPlaceholder.classList.remove('hidden');
             }
         }


        async function startWebcamStreamAndProcessing() {
             if (!selectedCameraId) {
                 throw new Error("No camera selected to start stream.");
             }
             console.info("Starting webcam stream for detection...");
             toggleOverlay('input', true, 'Starting camera...');
             stopWebcamStream(); // Ensure previous stream is stopped

             const constraints = { video: { deviceId: { exact: selectedCameraId } } };
             try {
                 mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
                 inputFeed.srcObject = mediaStream;
                 inputFeed.style.display = 'block';
                 inputPlaceholder.classList.add('hidden');
                 toggleOverlay('input', false); // Hide input overlay once stream starts
                 console.log("Webcam stream acquired for processing.");

                 // Wait briefly for video dimensions to become available
                  await new Promise(resolve => {
                      inputFeed.onloadedmetadata = resolve;
                      setTimeout(resolve, 500); // Failsafe timeout
                  });

                 if (inputFeed.videoWidth > 0 && inputFeed.videoHeight > 0) {
                      console.log(`Input feed resolution: ${inputFeed.videoWidth}x${inputFeed.videoHeight}`);
                      // Start the frame processing loop
                      startFrameProcessingLoop();
                 } else {
                      throw new Error("Webcam started but video dimensions are not available.");
                 }

             } catch (error) {
                  console.error("Fatal error starting webcam stream:", error);
                  toggleOverlay('input', true, `Camera Error: ${error.message}`); // Show error on overlay
                  stopWebcamStream(); // Clean up stream if partially started
                   inputFeed.style.display = 'none';
                   inputPlaceholder.classList.remove('hidden');
                  // Stop the backend detection process as well
                   await handleStopClick();
                   throw error; // Re-throw to be caught by handleStartClick
             }
         }


        function stopWebcamStream() {
            if (mediaStream) {
                console.log("Stopping webcam stream.");
                mediaStream.getTracks().forEach(track => track.stop());
                mediaStream = null;
                inputFeed.srcObject = null;
            }
             // Reset input feed display even if stream was null
             inputFeed.style.display = 'none';
             inputPlaceholder.classList.remove('hidden');
             toggleOverlay('input', false);
        }

        function startFrameProcessingLoop() {
            clearTimeout(frameProcessingTimeout); // Clear existing timeouts
            consecutiveErrors = 0; // Reset error count for the loop
            console.info("Starting frame processing loop...");
            toggleOverlay('output', true, 'Starting processing...'); // Show initial output overlay

            // Use a self-calling async function for the loop
            (async function loop() {
                 if (!isDetectionRunning) {
                     console.info("Detection stopped, exiting frame processing loop.");
                     toggleOverlay('output', false); // Ensure overlay is hidden on stop
                     return;
                 }

                 await processSingleFrame(); // Process one frame

                 // Schedule next iteration
                 if (isDetectionRunning) {
                      const delay = (consecutiveErrors > 0)
                                     ? Math.min(retryDelayBase * Math.pow(1.5, consecutiveErrors), 5000) // Slower backoff
                                     : frameInterval;
                     frameProcessingTimeout = setTimeout(loop, delay);
                 }
            })();
        }

        function stopFrameProcessingLoop() {
            console.info("Stopping frame processing loop.");
            clearTimeout(frameProcessingTimeout);
            frameProcessingTimeout = null;
             toggleOverlay('output', false); // Hide output overlay when stopped
             outputFeed.style.display = 'none'; // Hide output image
             outputPlaceholder.classList.remove('hidden'); // Show output placeholder
        }


        async function processSingleFrame() {
            // 1. Check readiness (Input feed should be running)
             if (!mediaStream || inputFeed.readyState < 2 || inputFeed.videoWidth === 0) {
                 console.warn(`processSingleFrame: Input stream not ready. State: ${inputFeed.readyState}, Dims: ${inputFeed.videoWidth}x${inputFeed.videoHeight}. Waiting...`);
                 consecutiveErrors++; // Count this as an error
                 toggleOverlay('output', true, 'Waiting for input feed...');
                 if (consecutiveErrors > maxConsecutiveErrors) await handleStopClick();
                 return; // Loop will retry
             }
              toggleOverlay('output', true, 'Processing...'); // Show processing message

            // 2. Capture Frame
            let frameData;
            try {
                const canvas = document.createElement('canvas');
                // Use a consistent processing size or scale down for performance
                const scaleFactor = 0.7; // Example: Process at 70% resolution
                canvas.width = inputFeed.videoWidth * scaleFactor;
                canvas.height = inputFeed.videoHeight * scaleFactor;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(inputFeed, 0, 0, canvas.width, canvas.height);
                frameData = canvas.toDataURL('image/jpeg', 0.7); // Quality 70%

                 if (!frameData || frameData.length < 150) throw new Error("Generated invalid frame data.");

            } catch (captureError) {
                console.error("Error capturing frame:", captureError);
                consecutiveErrors++;
                 toggleOverlay('output', true, 'Frame capture error...');
                if (consecutiveErrors > maxConsecutiveErrors) await handleStopClick();
                return; // Loop will retry
            }

            // 3. Send Frame to Backend
            const controller = new AbortController();
            const fetchTimeoutId = setTimeout(() => controller.abort(), requestTimeout);

             try {
                const response = await fetch(PROCESS_FRAME_ENDPOINT, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ frame: frameData }),
                    signal: controller.signal
                });
                clearTimeout(fetchTimeoutId);

                if (!response.ok) {
                    let errorMsg = `Server error: ${response.status}`;
                    try { errorMsg = (await response.json()).error || errorMsg; } catch (e) {}
                    throw new Error(errorMsg);
                }

                const data = await response.json();
                if (!data || !data.processed_frame || !data.stats) {
                    throw new Error("Invalid response structure from server.");
                }

                // 4. Update Output Feed and Stats on Success
                outputFeed.src = data.processed_frame;
                outputFeed.style.display = 'block';
                outputPlaceholder.classList.add('hidden');
                updateStatsDisplay(data.stats);
                toggleOverlay('output', false); // Hide overlay on success
                consecutiveErrors = 0; // Reset errors

            } catch (fetchError) {
                 console.error(`Frame processing error: ${fetchError.name === 'AbortError' ? 'Request timed out' : fetchError.message}`);
                 consecutiveErrors++;
                 toggleOverlay('output', true, `Processing Error (${consecutiveErrors})...`);
                 if (consecutiveErrors > maxConsecutiveErrors) {
                      alert("Detection stopped due to persistent processing errors.");
                      await handleStopClick();
                 }
                 // Loop will retry with backoff delay
            }
        }

        // === Other Handlers ===
        function handleVideoSourceChange() {
             const source = videoSourceSelect.value;
             console.log(`Video source changed to: ${source}`);
             if (isDetectionRunning) {
                  alert("Cannot change video source while detection is running. Please stop first.");
                  videoSourceSelect.value = isDetectionRunning ? (mediaStream ? 'webcam' : 'file') : source; // Revert selection
                  return;
             }
             stopWebcamStream(); // Stop webcam if active
             videoFileInput.value = ''; // Reset file input
             fileNameDisplay.textContent = 'No file selected';

             webcamControls.style.display = (source === 'webcam') ? 'block' : 'none';
             fileControls.style.display = (source === 'file') ? 'block' : 'none';
             updateUIForStoppedState(); // Ensure UI is reset
         }

        function handleFileSelection(event) {
            const file = event.target.files[0];
            if (file) {
                if (!file.type.startsWith('video/')) {
                     alert("Invalid file type. Please select a video file.");
                     videoFileInput.value = '';
                     fileNameDisplay.textContent = 'No file selected';
                     return;
                 }
                fileNameDisplay.textContent = file.name.length > 30 ? file.name.substring(0, 27) + '...' : file.name;
                 console.log(`File selected: ${file.name}, Size: ${file.size}, Type: ${file.type}`);
            } else {
                fileNameDisplay.textContent = 'No file selected';
            }
        }

        // === Camera Test Utility ===
        async function testCameraAccess() {
             console.info("Testing camera access...");
             toggleOverlay('input', true, 'Testing camera...');
             const testVideo = document.createElement('video');
             testVideo.autoplay = true; testVideo.muted = true; testVideo.playsinline = true;
             testVideo.style.cssText = "position:fixed; top:10px; left:10px; width:160px; height:120px; border:3px solid lime; z-index:10000; background:black;";
             document.body.appendChild(testVideo);
             let stream = null;
             try {
                 stream = await navigator.mediaDevices.getUserMedia({ video: true });
                 console.log("Test stream acquired.");
                 testVideo.srcObject = stream;
                 await testVideo.play(); // Ensure it plays
                 console.info("Camera test SUCCESSFUL.");
                 toggleOverlay('input', true, 'Test OK! Feed top-left.');
                 setTimeout(() => toggleOverlay('input', false), 3000); // Hide overlay after success message
             } catch (error) {
                 console.error("Camera test FAILED:", error.name, error.message);
                 alert(`Camera Test Failed: ${error.name}. Check permissions and ensure camera is not in use.`);
                 toggleOverlay('input', true, `Test Failed: ${error.name}`); // Show error
                 setTimeout(() => toggleOverlay('input', false), 4000); // Hide overlay after error
             } finally {
                 // Cleanup test elements after a delay
                 setTimeout(() => {
                     if (stream) stream.getTracks().forEach(track => track.stop());
                     if (testVideo.parentNode) testVideo.parentNode.removeChild(testVideo);
                     console.log("Test elements cleaned up.");
                 }, 4000);
             }
         }

    </script>
</body>
</html>

"""

    # Create templates directory if it doesn't exist
    # template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(TEMPLATE_DIR, exist_ok=True)

    # Write the HTML file
    template_path = os.path.join(TEMPLATE_DIR, 'index.html')
    try:
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Enhanced HTML template saved to {template_path}")
    except IOError as e:
        print(f"Error saving HTML template: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    # Ensure the detector module is available (real or dummy)
    if 'OvercrowdingDetector' not in globals():
        print("CRITICAL ERROR: OvercrowdingDetector class is not defined.")
        print("Make sure detector.py exists and is importable, or fix the dummy class definition.")
        exit(1)

    # Generate the HTML file on startup if it doesn't exist
    template_path = os.path.join(TEMPLATE_DIR, 'index.html')
    if not os.path.exists(template_path):
        save_html_template()
    else:
        print(f"Using existing template: {template_path}")
        # Optional: Force regeneration on every start for development
        # save_html_template()

    # Run the Flask app
    print("Starting Flask development server...")
    print(f"Access the application at http://localhost:5000 or http://<your-ip>:5000")
    # Use debug=True for development (auto-reload, detailed errors)
    # Use debug=False for production/deployment
    # Use threaded=True if your detector is I/O bound, False if CPU bound (or manage threads explicitly)
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)