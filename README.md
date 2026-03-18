# 🎯 Face Landmark Detection Service

[![Docker Build](https://github.com/mehaksandhudev/Advanced-Face-Detetction-service/actions/workflows/docker-build-push.yml/badge.svg)](https://github.com/mehaksandhudev/Advanced-Face-Detetction-service/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker Pulls](https://img.shields.io/docker/pulls/mehakxsandhu/face-detection-service)](https://hub.docker.com/r/mehakxsandhu/face-detection-service)

A fully local, production-ready, Dockerized face landmark detection API using **MediaPipe Face Mesh**. Provides **123 precisely mapped facial landmarks** with `x`, `y`, `z` coordinates. No cloud services, no API keys, 100% free and open source.

---

## 📑 Table of Contents
- [✨ Features](#-features)
- [📦 Disk Space](#-disk-space)
- [🚀 Quick Start](#-quick-start)
- [🔌 API Endpoints](#-api-endpoints)
- [🖼️ Image Detection](#-image-detection)
- [🎥 Video Detection](#-video-detection)
- [🗺️ Landmark Groups](#️-landmark-groups-123-points)
- [📊 Sample Response](#-sample-response)
- [🖥️ System Requirements](#️-system-requirements)
- [🔧 Configuration](#-configuration)
- [🚀 Deploy to Production](#-deploy-to-production)
- [🔄 Integration with Other Services](#-integration-with-other-services)
- [🧪 Local Testing](#-local-testing)
- [📁 Project Structure](#-project-structure)
- [🤝 Contributing & Support](#-contributing--support)
- [🛠️ Troubleshooting & FAQ](#️-troubleshooting--faq)
- [📄 License](#-license)

---

## ✨ Features

- **123 mapped landmarks** across 10 facial regions (mouth, lips, eyes, eyebrows, nose, jawline, forehead)
- **478 total landmarks** available via MediaPipe Face Mesh with iris refinement
- **Image & video support** — frame-by-frame video processing with configurable sampling
- **Multiple input methods** — file upload, URL, or local file path
- **CORS enabled** — works with browser-based apps and frontends
- **Production-ready** — gunicorn, health checks, error handling, security headers
- **CPU-optimized** — no GPU required, works on Intel HD Graphics

---

## 📦 Disk Space

| Component | Size |
|---|---|
| Docker image (total) | **~850 MB** |
| Python 3.11 slim base | ~150 MB |
| MediaPipe + OpenCV + deps | ~680 MB |
| Application code | < 1 MB |

---

## 🚀 Quick Start

**Option A: Pull from Docker Hub (Recommended)**
```bash
docker pull mehakxsandhu/face-detection-service:latest
docker run -d -p 5001:5001 --name face-detection mehakxsandhu/face-detection-service:latest
```

**Option B: Build from source**
```bash
git clone https://github.com/mehaksandhudev/Advanced-Face-Detetction-service.git
cd Advanced-Face-Detetction-service
docker-compose up --build -d
```

**Verify the container is running smoothly:**
```bash
curl http://localhost:5001/health
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API info and available endpoints |
| `GET` | `/health` | Health check |
| `GET` | `/landmarks/map` | Full landmark index mapping (JSON) |
| `POST` | `/detect` | Auto-detect image/video and process |
| `POST` | `/detect/image` | Image-only detection |
| `POST` | `/detect/video` | Video frame-by-frame detection |

---

## 🖼️ Image Detection

You have three ways to send images to the API depending on your use case:

**1. Direct File Upload (Multipart Form)**
Best for scripts, local processing, and direct API calls.
```bash
curl -X POST \
  http://localhost:5001/detect/image \
  -F "image=@/path/to/your/photo.jpg"
```

**2. Public URL (Google Vision API Compatible)**
Perfect for integrating with low-code tools like n8n, Make, or web frontends.
```bash
curl -X POST \
  http://localhost:5001/detect \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {
        "image": {
          "source": {
            "imageUri": "https://example.com/photo.jpg"
          }
        }
      }
    ]
  }'
```

**3. Local File Path (Mounted Volume)**
Best for processing large datasets already mounted securely inside the Docker container.
```bash
curl -X POST \
  http://localhost:5001/detect \
  -H "Content-Type: application/json" \
  -d '{
    "filePath": "/data/input/photo.jpg"
  }'
```

---

## 🎥 Video Detection

Process videos frame-by-frame. You can optimize server performance by skipping frames or setting a maximum frame limit.

```bash
curl -X POST \
  "http://localhost:5001/detect/video?sample_every_n=5&max_frames=100" \
  -F "image=@/path/to/your/video.mp4"
```

**Available Query Parameters:**
| Parameter | Default | Description |
|---|---|---|
| `max_frames` | `0` (all) | Maximum number of frames to process |
| `sample_every_n` | `1` (every frame) | Process every Nth frame (e.g., `5` = 20% of frames) |

---

## 🗺️ Landmark Groups (123 points)

The API naturally categorizes the 478 MediaPipe points into 10 highly useful JSON objects:

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

---

## 📊 Sample Response

All endpoints return a standardized, deeply structured JSON payload:

```json
{
  "faces_detected": 1,
  "faces": [
    {
      "face_index": 0,
      "confidence": 0.95,
      "bbox": { 
        "xmin": 120.5, "ymin": 80.3, "xmax": 380.2, "ymax": 420.8 
      },
      "landmarks_grouped": {
        "mouth_centers": {
          "upper_lip_inner_center": {
            "index": 13,
            "x": 245.3,
            "y": 310.5,
            "z": -12.4,
            "x_normalized": 0.384,
            "y_normalized": 0.647,
            "z_normalized": -0.019
          }
        },
        "left_eye": { 
          "iris_center": { "index": 468, "x": 210.1, "y": 195.3, "z": -8.2 } 
        }
      },
      "landmarks_all_478": [
        {"index": 0, "x": 245.8, "y": 302.1, "z": -10.5},
        {"index": 1, "x": 244.2, "y": 290.4, "z": -14.1}
      ],
      "total_landmarks": 478
    }
  ],
  "processing_time_sec": 0.108
}
```

---

## 🖥️ System Requirements

- **Docker Desktop** (Windows, macOS, or Linux)
- **CPU**: Any modern x86_64 (Intel i5+ or AMD equivalent)
- **RAM**: 4 GB available for the container
- **GPU**: Not required
- **Tested on:** Intel i7, 16 GB RAM, Intel HD Graphics

---

## 🔧 Configuration

**Environment Variables:**
| Variable | Default | Description |
|---|---|---|
| `PYTHONUNBUFFERED` | `1` | Set in docker-compose for real-time console logs |

**Container limits (in `docker-compose.yml`):**
- **Memory**: 4 GB max, 1 GB reserved limits applied
- **Restart policy**: `unless-stopped`

---

## 🚀 Deploy to Production

### Option 1: Docker Hub (CI/CD Automated)

This repository includes a GitHub Actions workflow that automatically builds and securely pushes to Docker Hub on every push to the `main` branch.

**Setup Instructions:**
1. Go to your GitHub repo → Settings → Secrets and variables → Actions
2. Add these repository secrets:
   - `DOCKERHUB_USERNAME` 
   - `DOCKERHUB_TOKEN` ([Create a Personal Access Token here](https://hub.docker.com/settings/security))
3. Push new code to `main` and watch GitHub Actions do the rest!

### Option 2: Manual Terminal Push
```bash
docker build -t mehakxsandhu/face-detection-service:latest .
docker push mehakxsandhu/face-detection-service:latest
```

---

## 🔄 Integration with Other Services

Add this snippet to your root `docker-compose.yml` to seamlessly connect the API to your existing stack:

```yaml
services:
  face-detection-service:
    image: mehakxsandhu/face-detection-service:latest
    container_name: face-detection-service
    ports:
      - "5001:5001"
    restart: unless-stopped
```

**Internal Container Access:** `http://face-detection-service:5001/detect`
**External Host Access:** `http://localhost:5001/detect`

---

## 🧪 Local Testing

A comprehensive test script is included to quickly verify all endpoints without manually writing `curl` commands.

```bash
# 1. Install the testing requirements
pip install requests

# 2. Run the full test suite against your local container
python test_request.py
```
This script will automatically verify the `/health`, `/landmarks/map`, and `/detect` endpoints against fallback checks and sample Google Vision API queries.

---

## 📁 Project Structure
```text
face_detection_service/
├── .github/workflows/     # CI/CD pipelines
│   └── docker-build-push.yml
├── app.py                 # Core Flask API (MediaPipe integration)
├── Dockerfile             # Python 3.11 slim + OpenCV headless
├── docker-compose.yml     # Container orchestration
├── requirements.txt       # Pinned production dependencies
├── test_request.py        # Automated API test suite
├── LANDMARKS_MAPPING.md   # Complete 123-point index mapping dictionary
├── LICENSE                # MIT License
├── .gitignore
├── .dockerignore
└── data/                  # Volume mount for processing huge local files
```

---

## 🤝 Contributing & Support

If you encounter a bug, have a feature request, or need help integrating the API:
1. Please check the existing [Issues](https://github.com/mehaksandhudev/Advanced-Face-Detetction-service/issues) page on GitHub.
2. If your problem isn't listed, open a new issue containing your Docker logs, the API request body, and expected output.
3. Pull requests are always welcome!

**Contact Me:**
- 📧 **Email:** `mehaksandhudev@gmail.com`
- 🌐 **Portfolio:** [www.mehak-sandhu.in](https://www.mehak-sandhu.in)

---

## 🛠️ Troubleshooting & FAQ

**1. Port 5001 is already in use** 
You can map it to any available port by modifying your Docker run command. For example, to run the service on port 8080: 
```bash
docker run -d -p 8080:5001 mehakxsandhu/face-detection-service:latest
```

**2. 413 Request Entity Too Large**
To prevent memory crashes, the API restricts direct file uploads to **16MB**. For larger video files, mount a local volume in your run command (`-v /my/videos:/data`) and pass the container's absolute file path via JSON instead of `multipart/form-data`.

**3. "Where do I track the XYZ facial feature?"**
Please consult the incredibly detailed [LANDMARKS_MAPPING.md](LANDMARKS_MAPPING.md) file. It contains the exact dictionary key names and MediaPipe indices for all 123 tracked points.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
