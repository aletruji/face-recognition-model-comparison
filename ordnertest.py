import os

# Pfad zu deinem LFW-Ordner
source_dir = r"C:\Users\aletr\Downloads\lfw-dataset\lfw-deepfunneled.zip\lfw-deepfunneled"

# Mindestanzahl an Bildern
min_images = 10

# Ergebnisliste
qualified_people = []

# Durch alle Unterordner (Personen)
for person in os.listdir(source_dir):
    person_path = os.path.join(source_dir, person)
    if os.path.isdir(person_path):
        images = [f for f in os.listdir(person_path) if f.lower().endswith(".jpg")]
        if len(images) >= min_images:
            qualified_people.append((person, len(images)))

# Ausgabe
print(f"\n✅ Personen mit mindestens {min_images} Bildern:")
for name, count in sorted(qualified_people, key=lambda x: -x[1]):
    print(f"{name}: {count} Bilder")

print(f"\nGesamt gefunden: {len(qualified_people)} Personen")
