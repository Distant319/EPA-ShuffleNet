import cv2
import numpy as np
import os
from pathlib import Path
import random

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

def adjust_brightness(img, factor):
    """
    Adjust image brightness.

    Args:
        img: input image
        factor: brightness factor in [0.5, 1.5]; 1.0 means no change
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] *= factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def adjust_contrast(img, factor):
    """
    Adjust image contrast.

    Args:
        img: input image
        factor: contrast factor in [0.5, 1.5]; 1.0 means no change
    """
    mean = np.mean(img)
    img_adj = (img - mean) * factor + mean
    return np.clip(img_adj, 0, 255).astype(np.uint8)

def color_jitter(img, hue_shift=10, saturation_factor=1.2, value_factor=1.1):
    """
    Color jittering: random hue, saturation, value.

    Args:
        img: input image
        hue_shift: hue offset in [-20, 20]
        saturation_factor: saturation multiplier in [0.8, 1.3]
        value_factor: value multiplier in [0.8, 1.3]
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    # hue
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    # saturation
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
    # value
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * value_factor, 0, 255)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def convert_to_grayscale(img):
    """
    Convert to 3-channel grayscale.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def apply_color_augmentations(input_folder, output_folder):
    """
    Apply multiple color augmentations to all images in a folder.

    Args:
        input_folder: path to input images
        output_folder: path to save augmented images
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
    print("=" * 70)

    for img_file in image_files:
        try:
            print(f"\nProcessing: {img_file.name}")
            img = imread_chinese(str(img_file))
            if img is None:
                print("  ✗ Failed to read image")
                continue

            base_name = img_file.stem

            # 1. original
            orig_path = os.path.join(output_folder, f"{base_name}_original.png")
            imwrite_chinese(orig_path, img)
            print(f"  ✓ Original: {base_name}_original.png")

            # 2. brightness
            bright_f = random.uniform(0.6, 1.4)
            img_b = adjust_brightness(img, bright_f)
            bright_path = os.path.join(output_folder,
                                       f"{base_name}_brightness_{bright_f:.2f}.png")
            imwrite_chinese(bright_path, img_b)
            print(f"  ✓ Brightness (factor={bright_f:.2f}): {os.path.basename(bright_path)}")

            # 3. contrast
            contrast_f = random.uniform(0.7, 1.5)
            img_c = adjust_contrast(img, contrast_f)
            contrast_path = os.path.join(output_folder,
                                         f"{base_name}_contrast_{contrast_f:.2f}.png")
            imwrite_chinese(contrast_path, img_c)
            print(f"  ✓ Contrast (factor={contrast_f:.2f}): {os.path.basename(contrast_path)}")

            # 4. color jitter
            hue = random.randint(-15, 15)
            sat_f = random.uniform(0.8, 1.3)
            val_f = random.uniform(0.8, 1.2)
            img_j = color_jitter(img, hue, sat_f, val_f)
            jitter_path = os.path.join(output_folder, f"{base_name}_jitter.png")
            imwrite_chinese(jitter_path, img_j)
            print(f"  ✓ Color jitter (hue={hue}, sat={sat_f:.2f}, val={val_f:.2f}): {base_name}_jitter.png")

            # 5. grayscale
            img_g = convert_to_grayscale(img)
            gray_path = os.path.join(output_folder, f"{base_name}_grayscale.png")
            imwrite_chinese(gray_path, img_g)
            print(f"  ✓ Grayscale: {base_name}_grayscale.png")

            # 6. combined: brightness + contrast
            bf2 = random.uniform(0.8, 1.2)
            cf2 = random.uniform(0.8, 1.2)
            img_comb = adjust_brightness(img, bf2)
            img_comb = adjust_contrast(img_comb, cf2)
            comb_path = os.path.join(output_folder, f"{base_name}_combined.png")
            imwrite_chinese(comb_path, img_comb)
            print(f"  ✓ Combined (bright={bf2:.2f}, contrast={cf2:.2f}): {base_name}_combined.png")

            # resize for comparison collage
            max_w = 400
            if img.shape[1] > max_w:
                scale = max_w / img.shape[1]
                new_size = (max_w, int(img.shape[0] * scale))
                img = cv2.resize(img, new_size)
                img_b = cv2.resize(img_b, new_size)
                img_c = cv2.resize(img_c, new_size)
                img_j = cv2.resize(img_j, new_size)
                img_g = cv2.resize(img_g, new_size)
                img_comb = cv2.resize(img_comb, new_size)

            # add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (0, 255, 0)

            def label(im, txt):
                im_l = im.copy()
                cv2.putText(im_l, txt, (10, 25), font, 0.5, color, 1)
                return im_l

            im1 = label(img, "Original")
            im2 = label(img_b, f"Brightness {bright_f:.2f}")
            im3 = label(img_c, f"Contrast {contrast_f:.2f}")
            im4 = label(img_j, "Color Jitter")
            im5 = label(img_g, "Grayscale")
            im6 = label(img_comb, "Combined")

            # 2×3 grid
            row1 = np.hstack([im1, im2, im3])
            row2 = np.hstack([im4, im5, im6])
            comparison = np.vstack([row1, row2])

            comp_path = os.path.join(output_folder, f"{base_name}_comparison.png")
            imwrite_chinese(comp_path, comparison)
            print(f"  ✓ Comparison grid: {base_name}_comparison.png")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"Done! Results saved to: {output_folder}")
    print("=" * 70)


if __name__ == "__main__":
    input_folder = "gray"          # change to your folder
    output_folder = "color_augmented"

    # random.seed(42)   # uncomment for reproducibility
    apply_color_augmentations(input_folder, output_folder)
