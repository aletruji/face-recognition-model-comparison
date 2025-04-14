import os
import numpy as np
import cv2
import dlib
from scipy.spatial import distance


class DlibResNetModel:

    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
        self.face_rec_model = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")
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
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                labels.append(label)
        return np.array(images), np.array(labels)

    def _get_embedding(self, img):
        dets = self.detector(img, 1)
        if len(dets) == 0:
            return None
        shape = self.shape_predictor(img, dets[0])
        return np.array(self.face_rec_model.compute_face_descriptor(img, shape))

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
        dists = [distance.euclidean(emb, ref_emb) for ref_emb in self.embeddings]
        min_idx = np.argmin(dists)
        return self.labels[min_idx], dists[min_idx]

    def get_label(self, label_id):
        return self.label_map.get(label_id, "Unknown")

    def save_model(self, path):
        # Dlib-Modell selbst ist pretrained und unveränderlich → kein Speichern nötig
        # Falls du Trainingsdaten (Embeddings + Labels) speichern willst, kannst du das hier ergänzen
        pass
