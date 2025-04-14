import os
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from sklearn.preprocessing import Normalizer
from scipy.spatial.distance import cosine

class ArcFaceModel:
    def __init__(self):
        self.model = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.model.prepare(ctx_id=0)
        self.l2_normalizer = Normalizer('l2')
        self.label_map = {}
        self.embeddings = []
        self.labels = []

    def load_data(self, dataset_path):
        images, labels = [], []
        persons = sorted(os.listdir(dataset_path))
        for label, person in enumerate(persons):
            self.label_map[label] = person
            person_path = os.path.join(dataset_path, person)
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, (160, 160))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                labels.append(label)
        return np.array(images), np.array(labels)

    def _get_embedding(self, img):
        faces = self.model.get(img)
        if not faces:
            return None
        emb = faces[0].embedding
        return self.l2_normalizer.transform([emb])[0]

    def train(self, X, y):
        self.embeddings = []
        self.labels = []
        for img, label in zip(X, y):
            emb = self._get_embedding(img)
            if emb is not None:
                self.embeddings.append(emb)
                self.labels.append(label)

    def predict(self, img):
        emb = self._get_embedding(img)
        if emb is None or not self.embeddings:
            return -1, 1.0
        dists = [cosine(emb, ref_emb) for ref_emb in self.embeddings]
        min_idx = np.argmin(dists)
        return self.labels[min_idx], dists[min_idx]

    def get_label(self, label_id):
        return self.label_map.get(label_id, "Unknown")

    def save_model(self, path):
        pass  # optional
