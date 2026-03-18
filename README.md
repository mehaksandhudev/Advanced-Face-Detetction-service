# 🎯 Face Landmark Detection Service

[![Docker Build](https://github.com/mehaksandhudev/Advanced-Face-Detetction-service/actions/workflows/docker-build-push.yml/badge.svg)](https://github.com/mehaksandhudev/Advanced-Face-Detetction-service/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker Pulls](https://img.shields.io/docker/pulls/mehakxsandhu/face-detection-service)](https://hub.docker.com/r/mehakxsandhu/face-detection-service)

A fully local, production-ready, Dockerized face landmark detection API using **MediaPipe Face Mesh**. Provides **123 precisely mapped facial landmarks** with x, y, z coordinates. No cloud services, no API keys, 100% free and open source.

## ✨ Features

- **123 mapped landmarks** across 10 facial regions (mouth, lips, eyes, eyebrows, nose, jawline, forehead)
- **478 total landmarks** available via MediaPipe Face Mesh with iris refinement
- **Image & video support** — frame-by-frame video processing with configurable sampling
- **Multiple input methods** — file upload, URL, or local file path
- **CORS enabled** — works with browser-based apps and frontends
- **Production-ready** — gunicorn, health checks, error handling, security headers
- **CPU-optimized** — no GPU required, works on Intel HD Graphics

## 📦 Disk Space

| Component | Size |
|---|---|
| Docker image (total) | **~850 MB** |
| Python 3.11 slim base | ~150 MB |
| MediaPipe + OpenCV + deps | ~680 MB |
| Application code | < 1 MB |

## 🚀 Quick Start

### Option A: Pull from Docker Hub
```bash
docker pull mehakxsandhu/face-detection-service:latest
docker run -d -p 5001:5001 --name face-detection mehakxsandhu/face-detection-service:latest
```

### Option B: Build from source
```bash
git clone https://github.com/mehaksandhudev/Advanced-Face-Detetction-service.git
cd Advanced-Face-Detetction-service
docker-compose up --build -d
```

### Verify it's running
```bash
curl http://localhost:5001/health
# {"status":"ok","service":"face-detection-service","version":"2.1.0"}
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API info and available endpoints |
| `GET` | `/health` | Health check |
| `GET` | `/landmarks/map` | Full landmark index mapping (JSON) |
| `POST` | `/detect` | Auto-detect image/video and process |
| `POST` | `/detect/image` | Image-only detection |
| `POST` | `/detect/video` | Video frame-by-frame detection |

### Image Detection
```bash
# File upload
curl -X POST -F "image=@photo.jpg" http://localhost:5001/detect/image

# URL-based (Google Vision API compatible)
curl -X POST -H "Content-Type: application/json" \
  -d '{"requests":[{"image":{"source":{"imageUri":"https://example.com/photo.jpg"}}}]}' \
  http://localhost:5001/detect

# Local file path (inside container)
curl -X POST -H "Content-Type: application/json" \
  -d '{"filePath":"/data/input/photo.jpg"}' \
  http://localhost:5001/detect
```

### Video Detection
```bash
# Process every 5th frame, max 100 frames
curl -X POST -F "image=@video.mp4" \
  "http://localhost:5001/detect/video?sample_every_n=5&max_frames=100"
```

| Query Param | Default | Description |
|---|---|---|
| `max_frames` | `0` (all) | Maximum frames to process |
| `sample_every_n` | `1` (every frame) | Process every Nth frame |

## 🗺️ Landmark Groups (123 points)

| Group | Points | Key Landmarks |
|---|---|---|
| `mouth_centers` | 4 | Upper/lower lip centers (inner + outer) |
| `lips_outer` | 19 | Full outer lip contour, corners at idx 61/291 |
| `lips_inner` | 19 | Full inner lip contour, corners at idx 78/308 |
| `left_eye` | 7 | Corners, lids, iris center (idx 468) |
| `right_eye` | 7 | Corners, lids, iris center (idx 473) |
| `left_eyebrow` | 10 | Inner edge, arch peak (idx 105), outer edge |
| `right_eyebrow` | 10 | Inner edge, arch peak (idx 334), outer edge |
| `nose` | 7 | Tip (idx 1), bridge, nostrils |
| `face_oval` | 36 | Full jawline contour, chin (idx 152) |
| `forehead` | 4 | Forehead reference points |

See [LANDMARKS_MAPPING.md](LANDMARKS_MAPPING.md) for the complete index-by-index reference.

## 📊 Sample Response

```json
{
  "faces_detected": 1,
  "faces": [{
    "face_index": 0,
    "confidence": 0.95,
    "bbox": { "xmin": 120.5, "ymin": 80.3, "xmax": 380.2, "ymax": 420.8 },
    "landmarks_grouped": {
      "mouth_centers": {
        "upper_lip_inner_center": {
          "index": 13, "x": 245.3, "y": 310.5, "z": -12.4,
          "x_normalized": 0.384, "y_normalized": 0.647, "z_normalized": -0.019
        }
      },
      "left_eye": { "iris_center": { "index": 468, "x": 210.1, "y": 195.3, "z": -8.2 } }
    },
    "landmarks_all_478": [{"index": 0, "x": 245.8, "y": 302.1, "z": -10.5}, ...],
    "total_landmarks": 478
  }],
  "processing_time_sec": 0.108
}
```

## 🖥️ System Requirements

- **Docker Desktop** (Windows, macOS, or Linux)
- **CPU**: Any modern x86_64 (Intel i5+ or AMD equivalent)
- **RAM**: 4 GB available for the container
- **GPU**: Not required
- Tested on: Intel i7, 16 GB RAM, Intel HD Graphics

## 🔧 Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `PYTHONUNBUFFERED` | `1` | Set in docker-compose for real-time logs |

Container limits (in `docker-compose.yml`):
- Memory: 4 GB max, 1 GB reserved
- Restart policy: `unless-stopped`

## 🚀 Deploy to Production

### Docker Hub (CI/CD)

This repo includes a GitHub Actions workflow that automatically builds and pushes to Docker Hub on every push to `main` or tagged release.

**Setup:**
1. Go to your GitHub repo → Settings → Secrets and variables → Actions
2. Add these secrets:
   - `DOCKERHUB_USERNAME` — your Docker Hub username
   - `DOCKERHUB_TOKEN` — a Docker Hub access token ([create one here](https://hub.docker.com/settings/security))
3. Push to `main` or create a tag like `v2.1.0`

### Manual Push
```bash
docker build -t mehakxsandhu/face-detection-service:latest .
docker push mehakxsandhu/face-detection-service:latest
```

## 🔄 Integration with Other Services

Add to an existing `docker-compose.yml`:
```yaml
  face-detection-service:
    image: mehakxsandhu/face-detection-service:latest
    container_name: face-detection-service
    ports:
      - "5001:5001"
    restart: unless-stopped
```

Access from other containers: `http://face-detection-service:5001/detect`
Access from host: `http://localhost:5001/detect`

## 📁 Project Structure
```
face_detection_service/
├── .github/workflows/     # CI/CD for Docker Hub
│   └── docker-build-push.yml
├── app.py                 # Flask API (123 mapped landmarks)
├── Dockerfile             # Python 3.11 slim + OpenCV headless
├── docker-compose.yml     # Container config (port 5001)
├── requirements.txt       # Pinned Python dependencies
├── test_request.py        # API test script
├── LANDMARKS_MAPPING.md   # Complete landmark index reference
├── LICENSE                # MIT License
├── .gitignore
├── .dockerignore
└── data/                  # Volume mount for file-based input
```

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
