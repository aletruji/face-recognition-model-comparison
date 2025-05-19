import os
import cv2
import numpy as np

# === Konfiguration ===
dataset_path = "input" #anpassen
output_path = "output" #anpassen
img_size = (100, 100)
num_components = 10  # wie viele Eigenfaces verwendet werden

# === Bilddaten laden ===
def load_images(path, size):
    images = []
    image_paths = []
    for person in sorted(os.listdir(path)):
        person_path = os.path.join(path, person)
        if not os.path.isdir(person_path):
            continue
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_resized = cv2.resize(img, size)
            images.append(img_resized.flatten())
            image_paths.append((person, img_name))
    return np.array(images), image_paths

# === Eigenfaces berechnen ===
def compute_eigenfaces(X, n_components):
    mean_face = np.mean(X, axis=0)
    X_centered = X - mean_face
    cov_matrix = np.cov(X_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx[:n_components]]
    return mean_face, eigvecs, X_centered

# === Hauptprogramm ===
if __name__ == "__main__":
    os.makedirs(output_path, exist_ok=True)
    X, paths = load_images(dataset_path, img_size)
    mean_face, eigvecs, X_centered = compute_eigenfaces(X, num_components)

    for i, (vec, (person, name)) in enumerate(zip(X_centered, paths)):
        weights = vec @ eigvecs  # Projektion auf Eigenfaces
        reconstruction = mean_face + weights @ eigvecs.T  # Rückprojektion
        reconstruction_img = reconstruction.reshape(img_size)

        # Optional: Normalisierung für Anzeige
        reconstruction_img = np.clip(reconstruction_img, 0, 255).astype(np.uint8)

        # Speicherpfad
        person_dir = os.path.join(output_path, person)
        os.makedirs(person_dir, exist_ok=True)
        out_path = os.path.join(person_dir, f"reconstructed_{name}")
        cv2.imwrite(out_path, reconstruction_img)

    print(f"Rekonstruierte Bilder gespeichert in: {output_path}")
