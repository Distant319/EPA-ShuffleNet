import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def convert_to_grayscale(image_path, output_dir='grayscale_output'):
    """
    Convert an image to grayscale and save the results.

    Args:
        image_path: path to the input image
        output_dir: directory where outputs will be saved
    """
    # Create output folder if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: unable to read image {image_path}")
        return

    # Method 1: standard OpenCV conversion
    gray_opencv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Method 2: weighted average (ITU-R BT.601)
    # Gray = 0.299*R + 0.587*G + 0.114*B
    b, g, r = cv2.split(image)
    gray_weighted = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)

    # Method 3: simple average
    gray_average = np.mean(image, axis=2).astype(np.uint8)

    # File name without extension
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Save grayscale images
    out_opencv = os.path.join(output_dir, f"{base_name}_grayscale.png")
    out_weighted = os.path.join(output_dir, f"{base_name}_grayscale_weighted.png")
    out_average = os.path.join(output_dir, f"{base_name}_grayscale_average.png")

    cv2.imwrite(out_opencv, gray_opencv)
    cv2.imwrite(out_weighted, gray_weighted)
    cv2.imwrite(out_average, gray_average)

    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Grayscale Conversion Comparison', fontsize=16, fontweight='bold')

    # Original
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original Color Image', fontsize=12)
    axes[0, 0].axis('off')

    # OpenCV standard
    axes[0, 1].imshow(gray_opencv, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title('Standard Grayscale (OpenCV)', fontsize=12)
    axes[0, 1].axis('off')

    # Weighted average
    axes[1, 0].imshow(gray_weighted, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title('Weighted Average (0.299R+0.587G+0.114B)', fontsize=12)
    axes[1, 0].axis('off')

    # Simple average
    axes[1, 1].imshow(gray_average, cmap='gray', vmin=0, vmax=255)
    axes[1, 1].set_title('Simple Average ((R+G+B)/3)', fontsize=12)
    axes[1, 1].axis('off')

    plt.tight_layout()
    comparison_path = os.path.join(output_dir, f"{base_name}_grayscale_comparison.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"Comparison figure saved: {comparison_path}")

    # Histogram comparison
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    fig2.suptitle('Grayscale Histogram Comparison', fontsize=16, fontweight='bold')

    for ax, data, title in zip(axes2,
                               [gray_opencv, gray_weighted, gray_average],
                               ['Standard Grayscale Histogram',
                                'Weighted Average Grayscale Histogram',
                                'Simple Average Grayscale Histogram']):
        ax.hist(data.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    hist_path = os.path.join(output_dir, f"{base_name}_histogram_comparison.png")
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    print(f"Histogram figure saved: {hist_path}")

    # Statistics
    print(f"\n=== Grayscale Statistics ===")
    print(f"Original image: {image_path}")
    print(f"Image size: {image.shape[1]} x {image.shape[0]}")

    for name, gray in zip(("Standard (OpenCV)", "Weighted Average", "Simple Average"),
                          (gray_opencv, gray_weighted, gray_average)):
        print(f"\n{name}:")
        print(f"  Mean: {np.mean(gray):.2f}")
        print(f"  Min:  {np.min(gray)}")
        print(f"  Max:  {np.max(gray)}")
        print(f"  Std:  {np.std(gray):.2f}")

    print(f"\nAll files saved to: {output_dir}")
    print("Files include:")
    print(f"  - {base_name}_grayscale.png")
    print(f"  - {base_name}_grayscale_weighted.png")
    print(f"  - {base_name}_grayscale_average.png")
    print(f"  - {base_name}_grayscale_comparison.png")
    print(f"  - {base_name}_histogram_comparison.png")

    plt.show()

if __name__ == "__main__":
    image_path = r"picture\RGB.png"   # change if necessary
    convert_to_grayscale(image_path)