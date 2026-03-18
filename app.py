"""
Face Landmark Detection Service v2.1.0
========================================
A production-ready, Dockerized face landmark detection API using MediaPipe Face Mesh.
Provides 123 precisely mapped facial landmarks with x, y, z coordinates.

Endpoints:
    POST /detect          - Detect faces in an image or video (auto-detect type)
    POST /detect/image    - Image-only detection
    POST /detect/video    - Video frame-by-frame detection
    GET  /health          - Health check
    GET  /landmarks/map   - Return the full landmark index mapping

GitHub:  https://github.com/mehaksandhudev/Advanced-Face-Detetction-service
Docker:  docker pull mehakxsandhu/face-detection-service
License: MIT
"""

import cv2
import mediapipe as mp
import numpy as np
import requests
import tempfile
import os
import json
import logging
import time
from flask import Flask, request, jsonify

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("face-detection-service")

VERSION = "2.1.0"

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Production Configuration
# ---------------------------------------------------------------------------
# Max upload size: 500 MB (for large videos)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

# CORS: Allow all origins (public API)
@app.after_request
def add_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Service-Version"] = VERSION
    return response

# Handle CORS preflight
@app.route("/", defaults={"path": ""}, methods=["OPTIONS"])
@app.route("/<path:path>", methods=["OPTIONS"])
def options_handler(path):
    return "", 204

# Handle 413 (file too large)
@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 500 MB."}), 413

# Handle 404
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found.", "available_endpoints": ["/health", "/detect", "/detect/image", "/detect/video", "/landmarks/map"]}), 404

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh

# ---------------------------------------------------------------------------
# LANDMARK MAPPING — 80+ points across all facial regions
# ---------------------------------------------------------------------------
# Each key maps to a dict of { sub-feature-name: landmark_index }.
# This gives you both the group AND the semantic meaning of every single point.

LANDMARK_GROUPS = {
    # ── Mouth Centers ──────────────────────────────────────────────────────
    "mouth_centers": {
        "upper_lip_inner_center":  13,
        "lower_lip_inner_center":  14,
        "upper_lip_outer_center":   0,
        "lower_lip_outer_center":  17,
    },

    # ── Outer Lips (19 points, full contour clockwise) ─────────────────────
    "lips_outer": {
        "left_corner":        61,
        "upper_left_1":      146,
        "upper_left_2":       91,
        "upper_left_3":      181,
        "upper_left_4":       84,
        "top_center":         17,
        "upper_right_4":     314,
        "upper_right_3":     405,
        "upper_right_2":     321,
        "upper_right_1":     375,
        "right_corner":      291,
        "lower_right_1":     409,
        "lower_right_2":     270,
        "lower_right_3":     269,
        "lower_right_4":     267,
        "bottom_center":       0,
        "lower_left_4":       37,
        "lower_left_3":       39,
        "lower_left_2":       40,
    },

    # ── Inner Lips (19 points, full contour clockwise) ─────────────────────
    "lips_inner": {
        "left_corner":        78,
        "upper_left_1":       95,
        "upper_left_2":       88,
        "upper_left_3":      178,
        "upper_left_4":       87,
        "top_center":         14,
        "upper_right_4":     317,
        "upper_right_3":     402,
        "upper_right_2":     318,
        "upper_right_1":     324,
        "right_corner":      308,
        "lower_right_1":     415,
        "lower_right_2":     310,
        "lower_right_3":     311,
        "lower_right_4":     312,
        "bottom_center":      13,
        "lower_left_4":       82,
        "lower_left_3":       81,
        "lower_left_2":       80,
    },

    # ── Left Eye (First Eye) ──────────────────────────────────────────────
    "left_eye": {
        "outer_corner":       33,
        "upper_lid_1":       160,
        "upper_lid_2":       158,
        "inner_corner":      133,
        "lower_lid_1":       153,
        "lower_lid_2":       144,
        "iris_center":       468,   # requires refine_landmarks=True
    },

    # ── Right Eye (Second Eye) ────────────────────────────────────────────
    "right_eye": {
        "inner_corner":      362,
        "upper_lid_1":       385,
        "upper_lid_2":       387,
        "outer_corner":      263,
        "lower_lid_1":       373,
        "lower_lid_2":       380,
        "iris_center":       473,   # requires refine_landmarks=True
    },

    # ── Left Eyebrow (10 points) ──────────────────────────────────────────
    "left_eyebrow": {
        "outer_1":            70,
        "outer_2":            63,
        "arch_peak":         105,
        "inner_1":            66,
        "inner_2":           107,
        "inner_edge_1":       55,
        "inner_edge_2":       65,
        "lower_1":            52,
        "lower_2":            53,
        "lower_3":            46,
    },

    # ── Right Eyebrow (10 points) ─────────────────────────────────────────
    "right_eyebrow": {
        "outer_1":           336,
        "outer_2":           296,
        "arch_peak":         334,
        "inner_1":           293,
        "inner_2":           300,
        "inner_edge_1":      285,
        "inner_edge_2":      295,
        "lower_1":           282,
        "lower_2":           283,
        "lower_3":           276,
    },

    # ── Nose ──────────────────────────────────────────────────────────────
    "nose": {
        "tip":                 1,
        "bridge_top":        168,
        "bridge_mid":          6,
        "left_nostril":      129,
        "right_nostril":     358,
        "left_alar":          98,
        "right_alar":        327,
    },

    # ── Face Oval / Jawline (36 points) ───────────────────────────────────
    "face_oval": {
        "jaw_01":             10,
        "jaw_02":            338,
        "jaw_03":            297,
        "jaw_04":            332,
        "jaw_05":            284,
        "jaw_06":            251,
        "jaw_07":            389,
        "jaw_08":            356,
        "jaw_09":            454,
        "jaw_10":            323,
        "jaw_11":            361,
        "jaw_12":            288,
        "jaw_13":            397,
        "jaw_14":            365,
        "jaw_15":            379,
        "jaw_16":            378,
        "jaw_17":            400,
        "jaw_18":            377,
        "chin":              152,
        "jaw_20":            148,
        "jaw_21":            176,
        "jaw_22":            149,
        "jaw_23":            150,
        "jaw_24":            136,
        "jaw_25":            172,
        "jaw_26":             58,
        "jaw_27":            132,
        "jaw_28":             93,
        "jaw_29":            234,
        "jaw_30":            127,
        "jaw_31":            162,
        "jaw_32":             21,
        "jaw_33":             54,
        "jaw_34":            103,
        "jaw_35":             67,
        "jaw_36":            109,
    },

    # ── Forehead / Top of Head References ─────────────────────────────────
    "forehead": {
        "center":             10,
        "left":              338,
        "right":             109,
        "midpoint_between_eyes": 168,
    },
}

# Flat lookup: { index: (group_name, sub_feature_name) }
INDEX_TO_FEATURE = {}
for group_name, features in LANDMARK_GROUPS.items():
    for feature_name, idx in features.items():
        INDEX_TO_FEATURE[idx] = (group_name, feature_name)


# ---------------------------------------------------------------------------
# Core Processing Functions
# ---------------------------------------------------------------------------

def create_face_mesh(static_mode=True, max_faces=10):
    """Create a MediaPipe FaceMesh instance with optimal settings."""
    return mp_face_mesh.FaceMesh(
        static_image_mode=static_mode,
        max_num_faces=max_faces,
        refine_landmarks=True,       # enables iris landmarks (468-477)
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def extract_landmarks(face_landmarks, width, height):
    """
    Extract ALL mapped landmarks from a single face.
    Returns a structured dict grouped by feature.
    """
    result = {}

    for group_name, features in LANDMARK_GROUPS.items():
        group_data = {}
        for feature_name, idx in features.items():
            if idx < len(face_landmarks.landmark):
                lm = face_landmarks.landmark[idx]
                group_data[feature_name] = {
                    "index": idx,
                    "x": round(lm.x * width, 2),
                    "y": round(lm.y * height, 2),
                    "z": round(lm.z * width, 2),     # z scaled by width (depth)
                    "x_normalized": round(lm.x, 6),
                    "y_normalized": round(lm.y, 6),
                    "z_normalized": round(lm.z, 6),
                }
        result[group_name] = group_data

    return result


def extract_all_478_landmarks(face_landmarks, width, height):
    """
    Extract ALL 478 raw landmarks (for users who need the complete mesh).
    Returns a list of { index, x, y, z, feature_group?, feature_name? }.
    """
    all_lm = []
    for idx, lm in enumerate(face_landmarks.landmark):
        entry = {
            "index": idx,
            "x": round(lm.x * width, 2),
            "y": round(lm.y * height, 2),
            "z": round(lm.z * width, 2),
        }
        if idx in INDEX_TO_FEATURE:
            entry["feature_group"] = INDEX_TO_FEATURE[idx][0]
            entry["feature_name"] = INDEX_TO_FEATURE[idx][1]
        all_lm.append(entry)
    return all_lm


def get_bounding_box(face_landmarks, width, height):
    """Calculate a tight bounding box from all mesh points."""
    xs = [lm.x for lm in face_landmarks.landmark]
    ys = [lm.y for lm in face_landmarks.landmark]

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    return {
        "xmin": round(xmin * width, 2),
        "ymin": round(ymin * height, 2),
        "xmax": round(xmax * width, 2),
        "ymax": round(ymax * height, 2),
        "width": round((xmax - xmin) * width, 2),
        "height": round((ymax - ymin) * height, 2),
    }


def process_single_image(image):
    """
    Process one image (numpy array, BGR).
    Returns list of face dicts.
    """
    height, width, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with create_face_mesh(static_mode=True) as mesh:
        results = mesh.process(image_rgb)

    faces = []
    if results.multi_face_landmarks:
        for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
            faces.append({
                "face_index": face_idx,
                "bbox": get_bounding_box(face_landmarks, width, height),
                "landmarks_grouped": extract_landmarks(face_landmarks, width, height),
                "landmarks_all_478": extract_all_478_landmarks(face_landmarks, width, height),
                "total_landmarks": len(face_landmarks.landmark),
                "confidence": 0.95,
            })
    return faces, {"width": width, "height": height}


def process_video_frames(video_path, max_frames=0, sample_every_n=1):
    """
    Process a video file frame by frame.

    Args:
        video_path:      path to the video file
        max_frames:      max frames to process (0 = all)
        sample_every_n:  process every Nth frame (1 = every frame)

    Returns:
        dict with metadata and per-frame results
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(f"Video: {total_frames} frames, {fps:.1f} FPS, {width}x{height}")

    frames_data = []
    frame_idx = 0
    processed_count = 0

    with create_face_mesh(static_mode=False, max_faces=10) as mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_every_n == 0:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = mesh.process(image_rgb)

                frame_faces = []
                if results.multi_face_landmarks:
                    for face_i, face_landmarks in enumerate(results.multi_face_landmarks):
                        frame_faces.append({
                            "face_index": face_i,
                            "bbox": get_bounding_box(face_landmarks, width, height),
                            "landmarks_grouped": extract_landmarks(face_landmarks, width, height),
                            "total_landmarks": len(face_landmarks.landmark),
                        })

                frames_data.append({
                    "frame_number": frame_idx,
                    "timestamp_sec": round(frame_idx / fps, 3),
                    "faces_detected": len(frame_faces),
                    "faces": frame_faces,
                })

                processed_count += 1
                if max_frames > 0 and processed_count >= max_frames:
                    break

            frame_idx += 1

    cap.release()

    return {
        "video_metadata": {
            "fps": fps,
            "total_frames": total_frames,
            "frames_processed": processed_count,
            "sample_every_n": sample_every_n,
            "resolution": {"width": width, "height": height},
            "duration_sec": round(total_frames / fps, 2),
        },
        "frames": frames_data,
    }


# ---------------------------------------------------------------------------
# Helper: download / save uploaded file to temp
# ---------------------------------------------------------------------------

def get_input_file(req):
    """
    Parse the incoming request and return (temp_file_path, [cleanup_paths]).
    Supports:
      1. JSON  { "requests": [{ "image": { "source": { "imageUri": "..." }}}] }
      2. JSON  { "filePath": "/data/input/photo.jpg" }
      3. Form  file upload  (field name: 'image' or 'file')
    """
    cleanup = []
    path = None

    # --- JSON body ---
    if req.is_json:
        data = req.get_json(silent=True) or {}

        # Google-Vision-style
        if "requests" in data:
            try:
                uri = data["requests"][0]["image"]["source"]["imageUri"]
                path, cleanup = _download(uri)
            except (KeyError, IndexError):
                pass

        # Simple filePath
        elif "filePath" in data:
            fp = data["filePath"]
            if os.path.isfile(fp):
                path = fp
            else:
                raise FileNotFoundError(f"filePath not found: {fp}")

    # --- File upload ---
    if path is None:
        for field in ("image", "file"):
            if field in req.files:
                f = req.files[field]
                ext = os.path.splitext(f.filename)[1] or ".tmp"
                tf = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                cleanup.append(tf.name)
                f.save(tf.name)
                tf.close()
                path = tf.name
                break

    return path, cleanup


def _download(url):
    """Download a URL to a temp file. Returns (path, [cleanup])."""
    suffix = ".tmp"
    lower = url.lower()
    for ext in (".mp4", ".avi", ".mov", ".mkv", ".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        if ext in lower:
            suffix = ext
            break

    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    logger.info(f"Downloading: {url}")
    with requests.get(url, stream=True, headers={"User-Agent": "Mozilla/5.0"}, timeout=60) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=8192):
            tf.write(chunk)
    tf.close()
    return tf.name, [tf.name]


def _is_video(path):
    """Guess whether a file is a video based on extension."""
    return os.path.splitext(path)[1].lower() in (
        ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v",
    )


def _cleanup(paths):
    for p in paths:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception as e:
            logger.warning(f"Cleanup failed for {p}: {e}")


# ---------------------------------------------------------------------------
# Flask Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "face-detection-service", "version": VERSION})


@app.route("/landmarks/map", methods=["GET"])
def landmarks_map():
    """Return the full landmark mapping as JSON so clients know what indices mean."""
    return jsonify({
        "total_mediapipe_landmarks": 478,
        "mapped_groups": {
            group: {name: idx for name, idx in features.items()}
            for group, features in LANDMARK_GROUPS.items()
        },
        "total_mapped_points": sum(len(v) for v in LANDMARK_GROUPS.values()),
    })


@app.route("/detect", methods=["POST"])
def detect():
    """
    Auto-detect whether input is image or video and process accordingly.
    Query params:
        max_frames     — for videos, limit the number of frames (default 0 = all)
        sample_every_n — for videos, process every Nth frame (default 1)
        include_all_478 — 'true' to include all 478 raw landmarks (default false for video)
    """
    cleanup = []
    try:
        source_path, cleanup = get_input_file(request)
        if not source_path:
            return jsonify({"error": "No valid input. Send a file upload (field 'image') or JSON body."}), 400

        if _is_video(source_path):
            return _handle_video(source_path, request)
        else:
            return _handle_image(source_path)

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"Error in /detect: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    finally:
        _cleanup(cleanup)


@app.route("/detect/image", methods=["POST"])
def detect_image():
    """Image-only endpoint."""
    cleanup = []
    try:
        source_path, cleanup = get_input_file(request)
        if not source_path:
            return jsonify({"error": "No valid input."}), 400
        return _handle_image(source_path)
    except Exception as e:
        logger.error(f"Error in /detect/image: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    finally:
        _cleanup(cleanup)


@app.route("/detect/video", methods=["POST"])
def detect_video():
    """Video-only endpoint with full frame-by-frame results."""
    cleanup = []
    try:
        source_path, cleanup = get_input_file(request)
        if not source_path:
            return jsonify({"error": "No valid input."}), 400
        return _handle_video(source_path, request)
    except Exception as e:
        logger.error(f"Error in /detect/video: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    finally:
        _cleanup(cleanup)


def _handle_image(source_path):
    """Process an image file and return JSON."""
    img = cv2.imread(source_path)
    if img is None:
        return jsonify({"error": "Could not read image file."}), 400

    t0 = time.time()
    faces, image_dim = process_single_image(img)
    elapsed = round(time.time() - t0, 3)

    return jsonify({
        "faces_detected": len(faces),
        "faces": faces,
        "primary_face": faces[0] if faces else None,
        "image_dimensions": image_dim,
        "processing_time_sec": elapsed,
    })


def _handle_video(source_path, req):
    """Process a video file frame-by-frame and return JSON."""
    max_frames = req.args.get("max_frames", 0, type=int)
    sample_every_n = req.args.get("sample_every_n", 1, type=int)
    sample_every_n = max(1, sample_every_n)

    t0 = time.time()
    result = process_video_frames(source_path, max_frames=max_frames, sample_every_n=sample_every_n)
    elapsed = round(time.time() - t0, 3)
    result["processing_time_sec"] = elapsed

    return jsonify(result)


# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "service": "face-detection-service",
        "version": VERSION,
        "endpoints": {
            "GET  /": "This help message",
            "GET  /health": "Health check",
            "GET  /landmarks/map": "Full landmark index mapping",
            "POST /detect": "Auto-detect image or video and return landmarks",
            "POST /detect/image": "Image-only face detection",
            "POST /detect/video": "Video frame-by-frame face detection",
        },
        "docs": "https://github.com/mehaksandhudev/Advanced-Face-Detetction-service",
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info(f"Starting Face Detection Service v{VERSION} on port 5001")
    app.run(host="0.0.0.0", port=5001, debug=False)
