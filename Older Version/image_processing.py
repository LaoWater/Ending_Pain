import cv2
import mediapipe as mp
import numpy as np
from main import stations_metrics, image_path, read_and_covert_rgb


def image_segmentation_mediapipe(img_path):
    # Initialize MediaPipe Selfie Segmentation.
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    image_rgb = read_and_covert_rgb(img_path)

    # Get the segmentation mask.
    results = selfie_segmentation.process(image_rgb)
    segmentation_mask = results.segmentation_mask

    # The mask contains values between 0 and 1, indicating how likely each pixel is to be part of the foreground.
    segmentation_mask = segmentation_mask > 0.5  # Convert to a binary mask.

    # Prepare a green background.
    background = np.zeros(image_rgb.shape, dtype=np.uint8)
    background[:] = [0, 255, 0]  # Green

    # Combine the original image with the green background based on the mask.
    foreground = np.where(segmentation_mask[:, :, None], image_rgb, background)

    return foreground


def color_difference_lab(color1, color2):
    """Calculate the Euclidean distance between two color vectors in Lab space."""
    return sum((a - b) ** 2 for a, b in zip(color1, color2)) ** 0.5


def find_upper_end_lab(image, start_x, start_y, tolerance=15):
    """Search upwards from the starting point in the Lab image to find where the color changes significantly."""
    mother_color = image[start_y, start_x]
    for y in range(start_y, max(-1, start_y - 200), -1):  # Search up to 200 pixels upwards.
        current_color = image[y, start_x]
        if color_difference_lab(current_color, mother_color) > tolerance:
            return start_x, y + 1  # Return the position just before the color change.
    return start_x, start_y  # If no change found, return the starting point.


# Display the image with foreground
# cv2.imshow('Foreground with Green Background', foreground_image_bgr)