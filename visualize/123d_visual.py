import face_recognition
import numpy as np
import matplotlib.pyplot as plt

# Bild laden
image_path = r"C:\Users\aletr\PycharmProjects\face-recognition-model-comparison\visualize\Condi_rice3.png"
image = face_recognition.load_image_file(image_path)

# Gesicht erkennen und Embedding berechnen
face_locations = face_recognition.face_locations(image)
face_encodings = face_recognition.face_encodings(image, face_locations)

if len(face_encodings) == 0:
    raise ValueError("Kein Gesicht erkannt im Bild!")

embedding = face_encodings[0]  # 128D-Vektor

# In 12x12-Matrix bringen (16 Felder bleiben leer)
matrix = np.full((12, 12), fill_value='', dtype=object)
embedding_values = np.round(embedding, 2)
matrix.flat[:128] = embedding_values  # Fülle nur die ersten 128 Zellen

# Darstellung als Tabelle (quadratisch)
fig, ax = plt.subplots(figsize=(8, 8))
ax.axis('off')
table = plt.table(cellText=matrix,
                  loc='center',
                  cellLoc='center',
                  colWidths=[0.07]*12)

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.2, 1.2)
plt.tight_layout()



plt.show()
