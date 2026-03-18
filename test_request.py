"""
Test script for the Face Detection Service API.
Run this AFTER the Docker container is running:
    docker-compose up --build

Usage:
    python test_request.py                          # test with a sample image URL
    python test_request.py --image photo.jpg        # test with a local image
    python test_request.py --video clip.mp4         # test with a local video
"""

import requests
import json
import sys
import argparse
import os

BASE_URL = "http://localhost:5001"


def test_health():
    """Test the health endpoint."""
    print("=" * 60)
    print("TEST: Health Check")
    print("=" * 60)
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"Status: {r.status_code}")
        print(f"Response: {json.dumps(r.json(), indent=2)}")
        return r.status_code == 200
    except requests.ConnectionError:
        print("ERROR: Cannot connect. Is the container running? (docker-compose up)")
        return False


def test_landmark_map():
    """Test the landmark map endpoint."""
    print("\n" + "=" * 60)
    print("TEST: Landmark Map")
    print("=" * 60)
    r = requests.get(f"{BASE_URL}/landmarks/map", timeout=10)
    data = r.json()
    print(f"Status: {r.status_code}")
    print(f"Total mapped points: {data.get('total_mapped_points', '?')}")
    print(f"Groups: {list(data.get('mapped_groups', {}).keys())}")
    return r.status_code == 200


def test_image_url(url):
    """Test with an image URL (Google Vision API style)."""
    print("\n" + "=" * 60)
    print(f"TEST: Image URL Detection")
    print("=" * 60)
    payload = {
        "requests": [{
            "image": {
                "source": {"imageUri": url}
            },
            "features": [{"type": "FACE_DETECTION"}]
        }]
    }
    r = requests.post(f"{BASE_URL}/detect", json=payload, timeout=60)
    data = r.json()
    print(f"Status: {r.status_code}")
    print(f"Faces detected: {data.get('faces_detected', 0)}")
    if data.get("primary_face"):
        pf = data["primary_face"]
        print(f"Primary face bbox: {json.dumps(pf['bbox'], indent=2)}")
        groups = pf.get("landmarks_grouped", {})
        for group_name in groups:
            print(f"  {group_name}: {len(groups[group_name])} points")
    print(f"Processing time: {data.get('processing_time_sec', '?')}s")
    return r.status_code == 200


def test_image_upload(filepath):
    """Test with a local image file upload."""
    print("\n" + "=" * 60)
    print(f"TEST: Image Upload — {filepath}")
    print("=" * 60)
    with open(filepath, "rb") as f:
        r = requests.post(f"{BASE_URL}/detect/image", files={"image": f}, timeout=60)
    data = r.json()
    print(f"Status: {r.status_code}")
    print(f"Faces detected: {data.get('faces_detected', 0)}")
    if data.get("faces"):
        for face in data["faces"]:
            print(f"\n  Face #{face['face_index']}:")
            groups = face.get("landmarks_grouped", {})
            for g, pts in groups.items():
                print(f"    {g}: {len(pts)} points")
                # Show first point as example
                first_key = next(iter(pts))
                first_pt = pts[first_key]
                print(f"      e.g. {first_key}: x={first_pt['x']}, y={first_pt['y']}, z={first_pt['z']}")
    print(f"Processing time: {data.get('processing_time_sec', '?')}s")
    return r.status_code == 200


def test_video_upload(filepath, max_frames=10, sample_every_n=5):
    """Test with a local video file upload."""
    print("\n" + "=" * 60)
    print(f"TEST: Video Upload — {filepath}")
    print(f"  max_frames={max_frames}, sample_every_n={sample_every_n}")
    print("=" * 60)
    with open(filepath, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/detect/video?max_frames={max_frames}&sample_every_n={sample_every_n}",
            files={"image": f},
            timeout=300
        )
    data = r.json()
    print(f"Status: {r.status_code}")
    meta = data.get("video_metadata", {})
    print(f"Video: {meta.get('total_frames')} total frames, {meta.get('fps')} FPS")
    print(f"Frames processed: {meta.get('frames_processed')}")

    frames = data.get("frames", [])
    if frames:
        print(f"\nFirst frame (#{frames[0]['frame_number']}, t={frames[0]['timestamp_sec']}s):")
        print(f"  Faces detected: {frames[0]['faces_detected']}")
        if frames[0].get("faces"):
            groups = frames[0]["faces"][0].get("landmarks_grouped", {})
            for g, pts in groups.items():
                print(f"    {g}: {len(pts)} points")

    print(f"\nProcessing time: {data.get('processing_time_sec', '?')}s")
    return r.status_code == 200


def main():
    parser = argparse.ArgumentParser(description="Test the Face Detection Service API")
    parser.add_argument("--image", help="Path to a local image file to test")
    parser.add_argument("--video", help="Path to a local video file to test")
    parser.add_argument("--max-frames", type=int, default=10, help="Max frames for video (default 10)")
    parser.add_argument("--sample-every", type=int, default=5, help="Sample every N frames (default 5)")
    args = parser.parse_args()

    # Always run health check first
    if not test_health():
        print("\nService is not running. Start it with: docker-compose up --build")
        sys.exit(1)

    test_landmark_map()

    # Test with provided files
    if args.image:
        if os.path.isfile(args.image):
            test_image_upload(args.image)
        else:
            print(f"File not found: {args.image}")

    if args.video:
        if os.path.isfile(args.video):
            test_video_upload(args.video, args.max_frames, args.sample_every)
        else:
            print(f"File not found: {args.video}")

    # If no files provided, test with a public sample image
    if not args.image and not args.video:
        print("\nNo local files provided. Testing with a public sample image URL...")
        test_image_url("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg")
        print("\nTip: Use --image or --video to test with your own files.")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
