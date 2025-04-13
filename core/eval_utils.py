import os
import tempfile


def measure_model_size(model):
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model_file")
        try:
            model.save_model(model_path)
        except Exception:
            return float('nan')

        if os.path.isdir(model_path):
            # Für TensorFlow SavedModel (Ordner)
            total_size = sum(os.path.getsize(os.path.join(dirpath, f))
                             for dirpath, _, files in os.walk(model_path)
                             for f in files)
        else:
            total_size = os.path.getsize(model_path)
        return total_size / 1024  # in KB
