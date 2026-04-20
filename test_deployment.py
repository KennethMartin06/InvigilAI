#!/usr/bin/env python3
"""
Quick deployment test script — validates backend health and inference.

Usage:
    python test_deployment.py <backend_url>

Example:
    python test_deployment.py https://invigilai-backend.railway.app
"""

import sys
import json
import requests
from typing import Optional

def test_health(base_url: str) -> bool:
    """Check /health endpoint."""
    url = f"{base_url}/health"
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()
        print(f"✓ /health: {r.status_code}")
        print(f"  - MLP loaded: {data.get('mlp_loaded')}")
        print(f"  - Scaler loaded: {data.get('scaler_loaded')}")
        print(f"  - Ensemble size: {data.get('ensemble_size')}")
        print(f"  - Models loaded: {data.get('models_loaded')}")
        return data.get("ensemble_size", 0) > 0
    except Exception as e:
        print(f"✗ /health failed: {e}")
        return False

def test_predict(base_url: str) -> bool:
    """Test /api/predict with dummy features."""
    url = f"{base_url}/api/predict"
    params = {
        "gaze_yaw": 5.0,
        "gaze_pitch": 10.0,
        "head_yaw": 2.0,
        "head_pitch": 3.0,
        "head_roll": 1.0,
        "face_count": 1.0,
        "gaze_deviation_ratio": 0.1,
        "face_embedding_norm": 0.95,
        "keystroke_rate": 4.0,
        "mean_dwell_time": 120.0,
        "mean_flight_time": 160.0,
        "burst_coefficient": 1.1,
        "cursor_velocity": 200.0,
        "click_frequency": 0.6,
        "idle_ratio": 0.15,
        "trajectory_linearity": 0.6,
    }
    try:
        r = requests.get(url, params=params, timeout=5)
        r.raise_for_status()
        data = r.json()
        print(f"✓ /api/predict: {r.status_code}")
        print(f"  - Predicted class: {data.get('class_name')}")
        print(f"  - Cheating probability: {data.get('cheating_probability')}")
        print(f"  - Is cheating: {data.get('is_cheating')}")
        print(f"  - Ensemble size: {data.get('ensemble_size')}")
        return True
    except Exception as e:
        print(f"✗ /api/predict failed: {e}")
        return False

def test_demo_login(base_url: str) -> Optional[str]:
    """Test demo login and return JWT token."""
    url = f"{base_url}/auth/login"
    payload = {
        "email": "admin@proctor.ai",
        "password": "admin123",
    }
    try:
        r = requests.post(url, json=payload, timeout=5)
        r.raise_for_status()
        data = r.json()
        token = data.get("access_token")
        print(f"✓ /auth/login: {r.status_code}")
        print(f"  - Token: {token[:20]}..." if token else "  - No token")
        return token
    except Exception as e:
        print(f"✗ /auth/login failed: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_deployment.py <backend_url>")
        print("Example: python test_deployment.py https://invigilai-backend.railway.app")
        sys.exit(1)

    base_url = sys.argv[1].rstrip("/")
    print(f"Testing {base_url}...\n")

    results = []
    results.append(("Health check", test_health(base_url)))
    print()
    results.append(("Predict endpoint", test_predict(base_url)))
    print()
    token = test_demo_login(base_url)
    results.append(("Demo login", token is not None))

    print("\n" + "="*50)
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"Summary: {passed}/{total} tests passed")
    if passed == total:
        print("✓ All checks passed — deployment is healthy!")
        sys.exit(0)
    else:
        print("✗ Some checks failed — review output above")
        sys.exit(1)

if __name__ == "__main__":
    main()
