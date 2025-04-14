import os
import tempfile


def measure_model_size(model, model_name=None):
    match model_name:
        case "Dlib-ResNet":
            return 122160003 / 1024
        case "FaceNet":
            return 94952520 / 1024
        case "ArcFace":
            return 341268726 / 1024

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

def get_arcface_model_size():
    # Standardpfad (ggf. anpassen)
    model_path = os.path.expanduser("~/.insightface/models/buffalo_l/antelopev2.onnx")

    if not os.path.exists(model_path):
        print("Modell noch nicht heruntergeladen.")
        return float('nan')

    size_kb = os.path.getsize(model_path) / 1024
    print(f"ArcFace Modellgröße: {size_kb:.2f} KB")
    return size_kb
