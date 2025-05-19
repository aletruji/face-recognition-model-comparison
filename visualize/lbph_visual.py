import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.feature import local_binary_pattern

# Parameter für LBP
radius = 1
n_points = 8 * radius
method = 'default'

# Pfade
input_path = r"C:\Users\aletr\Uni\lbph\Kofi_Annan_0003.jpg"
output_folder = r"C:\Users\aletr\Uni\lbph"
lbp_output_path = os.path.join(output_folder, "lbp_output.jpg")
hist_output_path = os.path.join(output_folder, "lbp_histogram.jpg")

# Bild laden
gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

# LBP berechnen
lbp = local_binary_pattern(gray, n_points, radius, method)

# LBP-Bild normalisieren und speichern
lbp_image = (lbp / lbp.max() * 255).astype(np.uint8)
cv2.imwrite(lbp_output_path, lbp_image)

# Histogramm berechnen
hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
hist = hist.astype("float")
hist /= hist.sum()

# Histogramm zeichnen und speichern
plt.figure(figsize=(8, 4))
plt.bar(range(len(hist)), hist, width=0.8, edgecolor='black')
plt.title("LBP Histogramm")
plt.xlabel("LBP-Werte")
plt.ylabel("Normierte Häufigkeit")
plt.tight_layout()
plt.savefig(hist_output_path, format='jpg')
plt.close()

print(f"LBP-Bild gespeichert unter: {lbp_output_path}")
print(f"Histogramm gespeichert unter: {hist_output_path}")
