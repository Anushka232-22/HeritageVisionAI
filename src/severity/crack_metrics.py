import cv2
import numpy as np


def extract_crack_metrics(image_path):

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Remove noise
    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Binary threshold
    _, thresh = cv2.threshold(
        blur,
        120,
        255,
        cv2.THRESH_BINARY_INV
    )

    # Crack pixel count
    crack_pixels = np.sum(thresh > 0)

    # Total image pixels
    total_pixels = thresh.shape[0] * thresh.shape[1]

    # Crack density
    density = crack_pixels / total_pixels

    # Find contours
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Crack area
    area = 0

    # Crack length
    length = 0

    for contour in contours:

        area += cv2.contourArea(contour)

        length += cv2.arcLength(
            contour,
            closed=False
        )

    return {

        "crack_length": round(float(length), 2),

        "crack_area": round(float(area), 2),

        "crack_density": round(float(density), 4)

    }


# For standalone testing
if __name__ == "__main__":

    image_path = "data/sample_images/crack1.jpg"

    result = extract_crack_metrics(image_path)

    print("\nCrack Metrics")
    print("=" * 30)

    for key, value in result.items():
        print(f"{key}: {value}")