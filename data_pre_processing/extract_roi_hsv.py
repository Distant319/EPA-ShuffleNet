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

def extract_red_pink_roi(input_folder, output_folder):
    """
    Extract red & pink ROIs via HSV color thresholds.

    Args:
        input_folder:  folder containing source images
        output_folder: folder where results will be saved
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
    print("=" * 60)

    # HSV ranges
    # Red wraps around 0° and 180°
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    # Pink
    lower_pink = np.array([150, 30, 100])
    upper_pink = np.array([170, 255, 255])

    print("HSV thresholds:")
    print("  Red-1:   H=[0-10],   S=[50-255], V=[50-255]")
    print("  Red-2:   H=[170-180], S=[50-255], V=[50-255]")
    print("  Pink:    H=[150-170], S=[30-255], V=[100-255]")
    print("=" * 60)

    for img_file in image_files:
        try:
            print(f"\nProcessing: {img_file.name}")
            img = imread_chinese(str(img_file))
            if img is None:
                print("  ✗ Failed to read image")
                continue

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Build masks
            mask_r1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask_r2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask_p = cv2.inRange(hsv, lower_pink, upper_pink)
            mask_comb = cv2.bitwise_or(mask_r1, mask_r2)
            mask_comb = cv2.bitwise_or(mask_comb, mask_p)

            # Morphology: remove noise & fill holes
            kernel_small = np.ones((3, 3), np.uint8)
            kernel_large = np.ones((15, 15), np.uint8)
            mask_clean = cv2.morphologyEx(mask_comb, cv2.MORPH_OPEN, kernel_small)
            mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_large)

            # Flood-fill to close large holes
            h, w = mask_clean.shape
            mask_ff = mask_clean.copy()
            mask_temp = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(mask_ff, mask_temp, (0, 0), 255)
            mask_clean = mask_clean | cv2.bitwise_not(mask_ff)

            # Extract ROI (keep original colours)
            roi = np.zeros_like(img)
            roi[mask_clean > 0] = img[mask_clean > 0]

            roi_ratio = (np.count_nonzero(mask_clean) / (h * w)) * 100
            print(f"  ✓ ROI area: {roi_ratio:.2f}%")

            base_name = img_file.stem

            # 1. raw mask (before hole-filling)
            mask_raw_path = os.path.join(output_folder, f"{base_name}_mask_original.png")
            imwrite_chinese(mask_raw_path, mask_comb)

            # 2. final mask
            mask_path = os.path.join(output_folder, f"{base_name}_mask.png")
            imwrite_chinese(mask_path, mask_clean)

            # 3. ROI image
            roi_path = os.path.join(output_folder, f"{base_name}_roi.png")
            imwrite_chinese(roi_path, roi)

            # 4. comparison collage
            img_ctr = img.copy()
            contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_ctr, contours, -1, (0, 0, 255), 2)

            max_w = 600
            if img.shape[1] > max_w:
                scale = max_w / img.shape[1]
                new_size = (max_w, int(img.shape[0] * scale))
                img = cv2.resize(img, new_size)
                img_ctr = cv2.resize(img_ctr, new_size)
                roi = cv2.resize(roi, new_size)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_color = (0, 255, 0)
            thick = 2

            im1 = img.copy()
            im2 = img_ctr.copy()
            im3 = roi.copy()
            cv2.putText(im1, "Original", (10, 30), font, 0.6, font_color, thick)
            cv2.putText(im2, "ROI Contours", (10, 30), font, 0.6, (0, 0, 255), thick)
            cv2.putText(im3, f"ROI ({roi_ratio:.1f}%)", (10, 30), font, 0.6, font_color, thick)

            comparison = np.hstack([im1, im2, im3])
            comp_path = os.path.join(output_folder, f"{base_name}_comparison.png")
            imwrite_chinese(comp_path, comparison)

            print(f"  ✓ Saved:")
            print(f"    - raw mask:     {base_name}_mask_original.png")
            print(f"    - final mask:   {base_name}_mask.png")
            print(f"    - ROI image:    {base_name}_roi.png")
            print(f"    - comparison:   {base_name}_comparison.png")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Finished! Results saved to: {output_folder}")
    print("=" * 60)


if __name__ == "__main__":
    input_folder = "New"
    output_folder = "roi_extracted"

    extract_red_pink_roi(input_folder, output_folder)
