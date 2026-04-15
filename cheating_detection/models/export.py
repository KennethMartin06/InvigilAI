"""
export.py -- Export trained models to ONNX format for production deployment.

Usage:
    python -m cheating_detection.models.export
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cheating_detection.config import (
    N_TOTAL_FEATURES,
    MLP_MODEL_PATH,
    ONNX_MODEL_PATH,
    MODELS_DIR,
)


def export_mlp_to_onnx(model=None, save_path=ONNX_MODEL_PATH, verbose=True):
    """Export the trained MLP model to ONNX format.

    Parameters
    ----------
    model : MLP instance (if None, loads from saved checkpoint)
    save_path : str -- destination .onnx file
    verbose : bool

    Returns
    -------
    str -- path to the exported ONNX file, or None if export failed
    """
    import torch

    try:
        import onnx
    except ImportError:
        if verbose:
            print("[ONNX] onnx not installed (pip install onnx), skipping export.")
        return None

    if model is None:
        from cheating_detection.models.train import MLP
        from cheating_detection.models.model_utils import load_pytorch_model
        try:
            model = load_pytorch_model(MLP, MLP_MODEL_PATH)
        except Exception as e:
            if verbose:
                print(f"[ONNX] Failed to load MLP model: {e}")
            return None

    model.eval()
    model.cpu()

    # Create dummy input matching expected feature count
    input_dim = N_TOTAL_FEATURES
    dummy_input = torch.randn(1, input_dim, dtype=torch.float32)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["features"],
            output_names=["logits"],
            dynamic_axes={
                "features": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
        )

        # Validate
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)

        if verbose:
            print(f"[ONNX] MLP exported -> {save_path}")
            file_size = os.path.getsize(save_path) / 1024
            print(f"[ONNX] Model size: {file_size:.1f} KB")

        return save_path

    except Exception as e:
        if verbose:
            print(f"[ONNX] Export failed: {e}")
        return None


def verify_onnx_model(onnx_path=ONNX_MODEL_PATH, verbose=True):
    """Verify ONNX model produces correct outputs by comparing with PyTorch."""
    import torch

    try:
        import onnxruntime as ort
    except ImportError:
        if verbose:
            print("[ONNX] onnxruntime not installed, skipping verification.")
        return False

    from cheating_detection.models.train import MLP
    from cheating_detection.models.model_utils import load_pytorch_model

    try:
        # Load PyTorch model
        pt_model = load_pytorch_model(MLP, MLP_MODEL_PATH)
        pt_model.eval()

        # Load ONNX model
        session = ort.InferenceSession(onnx_path)

        # Test with random input
        test_input = np.random.randn(5, N_TOTAL_FEATURES).astype(np.float32)

        # PyTorch prediction
        with torch.no_grad():
            pt_output = pt_model(torch.tensor(test_input)).numpy()

        # ONNX prediction
        onnx_output = session.run(None, {"features": test_input})[0]

        # Compare
        max_diff = np.max(np.abs(pt_output - onnx_output))
        if verbose:
            print(f"[ONNX] Max difference (PyTorch vs ONNX): {max_diff:.8f}")
            print(f"[ONNX] Verification: {'PASSED' if max_diff < 1e-5 else 'FAILED'}")

        return max_diff < 1e-5

    except Exception as e:
        if verbose:
            print(f"[ONNX] Verification failed: {e}")
        return False


if __name__ == "__main__":
    path = export_mlp_to_onnx()
    if path:
        verify_onnx_model(path)
