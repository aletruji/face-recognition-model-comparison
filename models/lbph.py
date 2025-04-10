import cv2
import os
import numpy as np

class LBPHModel:
    def __init__(self):
        self.model = cv2.face.LBPHFaceRecognizer_create()
        self.label_map = {}

    def load_data(self, dataset_path):
        images, labels = [], []
        persons = sorted(os.listdir(dataset_path))
        for label, person in enumerate(persons):
            self.label_map[label] = person
            person_path = os.path.join(dataset_path, person)
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (200, 200))
                images.append(img)
                labels.append(label)
        return np.array(images), np.array(labels)

    def train(self, X, y):
        self.model.train(X, y)

    def predict(self, img):
        return self.model.predict(img)

    def get_label(self, label_id):
        return self.label_map.get(label_id, "Unknown")
