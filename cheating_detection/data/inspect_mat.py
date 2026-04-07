"""Quick script to inspect MPIIGaze .mat file structure."""
import sys
import scipy.io as sio
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MAT_FILE = BASE_DIR / "datasets" / "gaze" / "MPIIGaze" / "Data" / "Normalized" / "p00" / "day01.mat"

print(f"Loading: {MAT_FILE}")
mat = sio.loadmat(str(MAT_FILE), squeeze_me=True)

print("\nTop-level keys:")
for k, v in mat.items():
    if k.startswith("__"):
        continue
    print(f"  '{k}': type={type(v).__name__}, ", end="")
    if hasattr(v, "shape"):
        print(f"shape={v.shape}, dtype={v.dtype}")
    elif hasattr(v, "__len__"):
        print(f"len={len(v)}")
    else:
        print(f"value={v}")

    # If struct, show sub-keys
    if hasattr(v, "dtype") and v.dtype.names:
        print(f"    Sub-keys: {v.dtype.names}")
        for subkey in v.dtype.names:
            sub = v[subkey]
            print(f"      '{subkey}': ", end="")
            try:
                arr = np.array(sub.tolist())
                print(f"shape={arr.shape}, dtype={arr.dtype}, sample={arr.flat[:3]}")
            except:
                print(f"type={type(sub)}")
