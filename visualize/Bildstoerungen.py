import cv2
import numpy as np

# Add Gaussian noise to the image
def add_noise(img, std_dev=55):
    noise = np.random.normal(0, std_dev, img.shape).astype(np.int16)
    img_int = img.astype(np.int16)

    # Add noise and clip to 0–255
    noisy_img = np.clip(img_int + noise, 0, 255).astype(np.uint8)
    return noisy_img

# Apply Gaussian blur
def add_blur(img):
    return cv2.GaussianBlur(img, (9, 9), 0)

# Darken the image by a given factor
def darken(img, factor=0.5):
    return cv2.convertScaleAbs(img, alpha=factor, beta=0)

# Apply all distortions to the image and save them
def process_image(path_to_image):
    # Read the original image
    img = cv2.imread(path_to_image)

    # Apply distortions
    blurred = add_blur(img)
    darkened = darken(img)
    noisy = add_noise(img)

    # Generate filenames
    base_name = path_to_image.rsplit('.', 1)[0]
    cv2.imwrite(f"{base_name}_blur.jpg", blurred)     # Save blurred image
    cv2.imwrite(f"{base_name}_dark.jpg", darkened)    # Save darkened image
    cv2.imwrite(f"{base_name}_noise.jpg", noisy)      # Save noisy image

# Example usage:
process_image("foto_link") #anpassen
