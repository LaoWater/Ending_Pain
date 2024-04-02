# ----------------------------------------------------------------------------------------------
# Ending Pain
#
# This tool is an image-processing socketing-bot for CQ online.
#
# Author :  Neo
# ----------------------------------------------------------------------------------------------
import cv2
import mediapipe as mp
import numpy as np
import math
from metrics_classes import PoseMetrics


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


# Define a function to draw 4x4 pixel squares on the given image at the specified center position.
def draw_square(image, center, color=(0, 0, 255), size=1):
    cv2.rectangle(image, (center[0] - size, center[1] - size), (center[0] + size, center[1] + size), color, -1)


def draw_grid(image, grid_size=8, color=(200, 200, 200)):  # Using light gray for the grid lines
    h, w, _ = image.shape
    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            cv2.rectangle(image, (x, y), (x + grid_size - 1, y + grid_size - 1), color,
                          1)  # Line thickness set to 1 for thin lines


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


def read_and_covert_rgb(img_path):
    # Processing image: reading, resizing, converting to RGB for Mediapipe processing
    image = cv2.imread(img_path)
    height, width, channels = image.shape
    image = cv2.resize(image, (510, 680))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image_rgb


def process_image_mediapipe(img_path):
    # Mediapipe initialization
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    image_rgb = read_and_covert_rgb(img_path)

    # Processing image with Mediapipe
    pose_results = pose.process(image_rgb)
    # Outputs RGB image
    mp_drawing.draw_landmarks(image_rgb, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    return pose_results, image_rgb


def image_blueprint():
    # Create a blueprint image with the same dimensions as the original, filled with white.
    blueprint_image = np.ones((680, 510, 3), dtype=np.uint8) * 255
    # Draw a 8x8 pixel grid on the blueprint image as the background.
    draw_grid(blueprint_image)

    # MediaPipe processing
    pose_results, image_rgb = process_image_mediapipe(image_path)
    height, width, _ = image_rgb.shape

    # Print all the mapped points
    index = 0
    print("Pose Results landmarks", pose_results.pose_landmarks.landmark)

    # Mapping nose for figure centering
    nose_landmark = pose_results.pose_landmarks.landmark[0]
    nose_x, nose_y = int(nose_landmark.x * width), int(nose_landmark.y * height)

    # Desired position (center of X axis, 20% of Y axis from top)
    desired_x = width // 2
    desired_y = int(height * 0.2)

    # Calculate shifts
    shift_x = desired_x - nose_x
    shift_y = desired_y - nose_y
    print("Calculated shifts (x,y)", shift_x, shift_y)

    for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
        height, width, _ = image_rgb.shape
        # Correctly applying shifts to move landmarks towards the top left
        cx, cy = int(landmark.x * width) + shift_x, int(landmark.y * height) + shift_y
        if index > 8 or index == 0:
            pose_metrics.update_landmark(index, (cx, cy))
            draw_square(blueprint_image, (cx, cy), color=(255, 0, 0))  # Draw on the blueprint image.
        # Shoulders
        if 10 < index < 13:
            pose_metrics.update_landmark(index, (cx, cy))
            # Convert image to Lab color space for more uniform color difference evaluation.
            draw_square(blueprint_image, (cx, cy - 70), color=(0, 0, 255))
        print(f"Landmark {idx}: ({cx}, {cy})")
        index += 1

    return blueprint_image


###################
# Starting Script #
###################

pose_metrics = PoseMetrics()
image_path = 'Photos/Anterior_view_1.JPG'

pose_data, image_mediapipe = process_image_mediapipe(image_path)
blueprint_processed_grid = image_blueprint()
foreground_image = image_segmentation_mediapipe(image_path)

# Convert the processed image back to BGR for displaying with OpenCV.
image_bgr = cv2.cvtColor(image_mediapipe, cv2.COLOR_RGB2BGR)
# Convert the processed image back to BGR for displaying with OpenCV.
foreground_image_bgr = cv2.cvtColor(foreground_image, cv2.COLOR_RGB2BGR)

# Display the original image with pose landmarks.
cv2.imshow('Output BGR', image_bgr)
# Display the blueprint grid.
cv2.imshow('Blueprint with Landmarks', blueprint_processed_grid)
# Display the image with foreground
# cv2.imshow('Foreground with Green Background', foreground_image_bgr)

print("Metrics:")
for part, metrics in pose_metrics.metrics.items():
    print(f'{part}:')
    for key, value in metrics.items():
        print(f'  {key}: {value}')
    print('\n', end='')  # Add an extra newline for separation between parts



cv2.waitKey(0)
cv2.destroyAllWindows()
