import os
import cv2
import numpy as np

class FisherfacesModel:
    def __init__(self):
        self.model = cv2.face.FisherFaceRecognizer_create()

    def train(self, X_train, y_train):
        self.model.train(X_train, np.array(y_train))

    def predict(self, img):
        label, confidence = self.model.predict(img)
        return label, confidence

    def save_model(self, path):
        self.model.save(path)

    def load_data(self, dataset_path):
        X = []
        y = []
        label_map = {}
        label_counter = 0

        for person_name in sorted(os.listdir(dataset_path)):
            person_path = os.path.join(dataset_path, person_name)
            if not os.path.isdir(person_path):
                continue

            if person_name not in label_map:
                label_map[person_name] = label_counter
                label_counter += 1

            label = label_map[person_name]

            for img_name in os.listdir(person_path):
                if not img_name.lower().endswith('.jpg'):
                    continue

                img_path = os.path.join(person_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    continue

                img_resized = cv2.resize(img, (100, 100))
                X.append(img_resized)
                y.append(label)

        return np.array(X), np.array(y)
