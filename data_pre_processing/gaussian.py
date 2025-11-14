import cv2
import numpy as np
import os
from pathlib import Path

def imread_chinese(file_path):
    """
    Read an image whose path contains Chinese characters.
    """
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)

def imwrite_chinese(file_path, img):
    """
    Save an image to a path that may contain Chinese characters.
    """
    cv2.imencode('.png', img)[1].tofile(file_path)

def apply_gaussian_filter(input_folder, output_folder, kernel_size=(5, 5), sigma=1.0):
    """
    Apply Gaussian filtering to every image in a folder.

    Args:
        input_folder:  folder containing source images
        output_folder: folder where filtered images will be saved
        kernel_size:   Gaussian kernel size (must be odd), default (5, 5)
        sigma:         Gaussian standard deviation, default 1.0
    """
    os.makedirs(output_folder, exist_ok=True)

    input_path = Path(input_folder)
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_files = [f for f in input_path.iterdir()
                   if f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"No images found in {input_folder}")
        return

    print(f"Found {len(image_files)} image(s)")
    print(f"Gaussian-filter params: kernel_size={kernel_size}, sigma={sigma}")
    print("-" * 50)

    for img_file in image_files:
        try:
            img = imread_chinese(str(img_file))
            if img is None:
                print(f"Failed to read: {img_file.name}")
                continue

            filtered = cv2.GaussianBlur(img, kernel_size, sigma)

            out_path = os.path.join(output_folder, f"gaussian_{img_file.name}")
            imwrite_chinese(out_path, filtered)
            print(f"✓ Processed: {img_file.name} -> {out_path}")

        except Exception as e:
            print(f"✗ Error on {img_file.name}: {e}")

    print("-" * 50)
    print(f"Done! Results saved to: {output_folder}")


if __name__ == "__main__":
    input_folder = "New"
    output_folder = "gaussian_filtered"

    # You can tune:
    # - kernel_size: larger -> more blur, must be odd, e.g. (3,3), (7,7), (9,9)
    # - sigma: larger -> more blur, e.g. 0.5, 1.5, 2.0
    apply_gaussian_filter(
        input_folder=input_folder,
        output_folder=output_folder,
        kernel_size=(5, 5),
        sigma=1.0
    )