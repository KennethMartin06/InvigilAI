"""Debug MPIIGaze .mat file access."""
import scipy.io as sio
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MAT_FILE = BASE_DIR / "datasets" / "gaze" / "MPIIGaze" / "Data" / "Normalized" / "p00" / "day01.mat"

mat = sio.loadmat(str(MAT_FILE), squeeze_me=True)
data = mat["data"]

print("data type:", type(data))
print("data dtype:", data.dtype)
print("data shape:", data.shape)

# Try accessing right eye
print("\nTrying data['right']...")
right = data["right"]
print("  right type:", type(right))
print("  right dtype:", right.dtype)
print("  right shape:", right.shape)

print("\nTrying data['right'].item()...")
right_item = data["right"].item()
print("  right_item type:", type(right_item))
if hasattr(right_item, "dtype"):
    print("  right_item dtype:", right_item.dtype)

print("\nTrying gaze access...")
try:
    gaze = right_item["gaze"]
    print("  gaze type:", type(gaze))
    print("  gaze shape:", np.array(gaze).shape)
    print("  gaze sample:", np.array(gaze)[:2])
except Exception as e:
    print("  FAILED:", e)

# Try alternative access
print("\nTrying direct data['right']['gaze']...")
try:
    gaze2 = data["right"]["gaze"]
    print("  type:", type(gaze2))
    arr = np.array(gaze2)
    print("  shape:", arr.shape)
except Exception as e:
    print("  FAILED:", e)
