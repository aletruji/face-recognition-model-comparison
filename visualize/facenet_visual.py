import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import cv2

# === Dein eigenes Modell verwenden ===
from models.FaceNet import FaceNetModel

# === Initialisiere das Modell ===
model = FaceNetModel()

# === Pfade ===
input_folder = r"C:\Users\aletr\PycharmProjects\face-recognition-model-comparison\data"
output_path = r"C:\Users\aletr\Uni\face\facenet_embeddings_pca.png"

# === Embeddings sammeln ===
embeddings = []
labels = []

persons = sorted(os.listdir(input_folder))
for index, folder in enumerate(persons, start=1):
    person_path = os.path.join(input_folder, folder)
    if not os.path.isdir(person_path):
        continue

    label = f"person{index:02d}"  # z. B. person01, person02 ...

    for image_name in sorted(os.listdir(person_path)):
        image_path = os.path.join(person_path, image_name)
        try:
            img = cv2.imread(image_path)
            if img is None:
                continue
            img = cv2.resize(img, (160, 160))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            embedding = model.model.embeddings([img])[0]
            embeddings.append(embedding)
            labels.append(label)
        except Exception as e:
            print(f"Fehler bei {image_path}: {e}")



embeddings = np.array(embeddings)

# === PCA-Projektion ===
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# === Plotten ===
plt.figure(figsize=(10, 7))
unique_labels = sorted(set(labels))
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

for i, label in enumerate(unique_labels):
    idxs = [j for j, l in enumerate(labels) if l == label]
    plt.scatter(embeddings_2d[idxs, 0], embeddings_2d[idxs, 1], label=label, alpha=0.8, color=colors[i])

plt.title("2D-Projektion der FaceNet-Embeddings (PCA)")
plt.xlabel("Hauptkomponente 1")
plt.ylabel("Hauptkomponente 2")
plt.legend(loc='best', fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.savefig(output_path)
plt.show()
