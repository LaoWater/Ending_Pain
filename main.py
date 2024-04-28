# ----------------------------------------------------------------------------------------------
# Ending Pain
#
# This tool is an image-processing tool for deep biomechanical readings and Recovery plan.
#
# Author :  Neo & N1ptic
# ----------------------------------------------------------------------------------------------
import json
import cv2
import mediapipe as mp
import numpy as np
import os
from metrics_classes_v2 import StationsMetrics, TracksMetrics
from tracks_processing import calculate_origin_nexus, calculate_nebula_nexus, calculate_horizon_nexus
from nexus_database import process_origin_nexus_dataset, process_nebula_nexus_dataset, process_horizon_nexus_dataset

working_dir = os.getcwd()
file_path = os.path.join(working_dir, r'json_formatted_results')


# Define a function to draw 4x4 pixel squares on the given image at the specified center position.
def draw_square(image, center, color=(0, 0, 255), size=1):
    cv2.rectangle(image, (center[0] - size, center[1] - size), (center[0] + size, center[1] + size), color, -1)


def draw_grid(image, grid_size=8, color=(200, 200, 200)):  # Using light gray for the grid lines
    h, w, _ = image.shape
    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            cv2.rectangle(image, (x, y), (x + grid_size - 1, y + grid_size - 1), color,
                          1)  # Line thickness set to 1 for thin lines


def read_and_covert_rgb(img_path):
    # Processing image: reading, resizing, converting to RGB for Mediapipe processing
    image = cv2.imread(img_path)
    # Consider Re-sizing to default value or using original image size.
    # height, width, channels = image.shape
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
    # print("Pose Results landmarks", pose_results.pose_landmarks.landmark)

    # Mapping nose for figure centering
    nose_landmark = pose_results.pose_landmarks.landmark[0]
    nose_x, nose_y = int(nose_landmark.x * width), int(nose_landmark.y * height)

    # Desired position (center of X axis, 20% of Y axis from top)
    desired_x = width // 2
    desired_y = int(height * 0.2)

    # Calculate shifts for centering image
    shift_x = desired_x - nose_x
    shift_y = desired_y - nose_y
    # print("Calculated shifts (x,y)", shift_x, shift_y)

    for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
        height, width, _ = image_rgb.shape
        # Correctly applying shifts to move landmarks towards the top left
        cx, cy = int(landmark.x * width) + shift_x, int(landmark.y * height) + shift_y
        if index > 8 or index == 0:
            stations_metrics.update_landmark(index, (cx, cy))
            draw_square(blueprint_image, (cx, cy), color=(255, 0, 0))  # Draw on the blueprint image.
        # Shoulders
        if 10 < index < 13:
            stations_metrics.update_landmark(index, (cx, cy))
            # Convert image to Lab color space for more uniform color difference evaluation.
            draw_square(blueprint_image, (cx, cy - 70), color=(0, 0, 255))
        # Left Shoulder test
        if index == 11:
            draw_square(blueprint_image, (cx, cy - 15), color=(45, 100, 99))
        # print(f"Landmark {idx}: ({cx}, {cy})")
        index += 1
    stations_metrics.calculate_tracks()

    return blueprint_image


def calculate_weight_distribution(tracks_metrics_instance):
    return tracks_metrics_instance.sum_of_lambdas()


def process_origin_nexus(tracks_metrics_class, stations_metrics_class):
    alpha_p = tracks_metrics_class.tracks['Lateral Line Proximal']['Alpha']
    alpha_i = tracks_metrics_class.tracks['Lateral Line Inferior']['Alpha']
    if alpha_p + alpha_i > 0:
        foot_rotation = stations_metrics_class.metrics['Feet']['Right Analysis']['Rotation Degree']
    else:
        foot_rotation = stations_metrics_class.metrics['Feet']['Left Analysis']['Rotation Degree']

    origin_nexus_output = calculate_origin_nexus(alpha_p, alpha_i, foot_rotation)
    print("\nOrigin Nexus Processing..\n"
          "(First Treatment Plan Blueprint - calculated from Proximal, Inferior LL. and foot rotations):\n",
          origin_nexus_output)
    return origin_nexus_output


def process_nebula_nexus(tracks_metrics_class):
    alpha_os = tracks_metrics_class.tracks['Oblique Sling']['Alpha']
    lambda_os = tracks_metrics_class.tracks['Oblique Sling']['Lambda']
    alpha_p = tracks_metrics_class.tracks['Lateral Line Proximal']['Alpha']
    nebula_nexus_output = calculate_nebula_nexus(alpha_os, alpha_p, lambda_os)
    print("\nNebula Nexus Processing..\n"
          "(Second Treatment Plan Blueprint - calculated from Oblique Sling and Proximal LL):\n",
          nebula_nexus_output)
    return nebula_nexus_output


def process_horizon_nexus(tracks_metrics_class, stations_metrics_class):
    alpha_p = tracks_metrics_class.tracks['Lateral Line Proximal']['Alpha']
    alpha_i = tracks_metrics_class.tracks['Lateral Line Inferior']['Alpha']
    if alpha_p + alpha_i > 0:
        lambda_shoulder_rotation = stations_metrics_class.metrics['Feet']['Right Analysis']['Rotation Degree']
    else:
        lambda_shoulder_rotation = stations_metrics_class.metrics['Feet']['Left Analysis']['Rotation Degree']
    alpha_s = tracks_metrics_class.tracks['Lateral Line Superior']['Alpha']
    alpha_arm_l = tracks_metrics_class.tracks['Arm Line']['Alpha']
    horizon_nexus_output = calculate_horizon_nexus(alpha_s, alpha_arm_l, lambda_shoulder_rotation)
    print("\nHorizon Nexus Processing..\n"
          "(Third Treatment Plan Blueprint - calculated from Lateral Line Superior and Arm Line):\n",
          horizon_nexus_output)
    return horizon_nexus_output


def print_metrics(stations_metrics_class, tracks_metrics_class):
    for i, j in stations_metrics_class.metrics.items():
        print(f"{i}; {j}")

    print("\n\nTrack Metrics: \n*Dictionary"
          "\n *Alpha = Length-Based Compression. \nPositive = Right LL. (Side) Compression"
          "\nNegative = Left LL. (Side) Compression\n\n"
          "*Lambda = Track angle with vertical vector\n"
          "Positive = Left Side Weight Distribution\n"
          "Negative = Right Side Weight Distribution\n")
    for track, details in tracks_metrics_class.tracks.items():
        print(f"{track}: {details}")

    total_weighted_l = tracks_metrics_class.sum_of_lambdas()
    print("\nCalculated Weight Distribution:", total_weighted_l)


def generate_json_results(results, nexus_type):
    global file_path
    path_adjacent = None
    # Write results to a file
    if nexus_type == 'origin':
        path_adjacent = r'\origin_nexus_results.json'
    if nexus_type == 'nebula':
        path_adjacent = r'\nebula_nexus_results.json'
    if nexus_type == 'horizon':
        path_adjacent = r'\horizon_nexus_results.json'

    final_path = file_path + path_adjacent
    with open(final_path, 'w') as file:
        json.dump(results, file)


###################
###################
# Starting Script #
###################
###################
stations_metrics = StationsMetrics()
image_path = r'C:\Users\baciu\Desktop\Neo\Ending_Pain\Photos\Anterior_view_6.JPG'

pose_data, image_mediapipe = process_image_mediapipe(image_path)
blueprint_processed_grid = image_blueprint()

# Updating Tracks class once Stations is complete.
tracks_metrics = TracksMetrics(stations_metrics)
tracks_metrics.calculate_tracks()
# Calculating Weight Distribution based on Lambdas (the higher to 1, the more weight is distributed to left side)
total_weighted_lambda = tracks_metrics.sum_of_lambdas()

# Printing Metrics
print_metrics(stations_metrics, tracks_metrics)

###############################################
# Nexus Processing ############################
# Start of data processing for treatment plan #
###############################################
origin_nexus_processed = process_origin_nexus(tracks_metrics, stations_metrics)
nebula_nexus_processed = process_nebula_nexus(tracks_metrics)
horizon_nexus_processed = process_horizon_nexus(tracks_metrics, stations_metrics)
print('\n\n\n')

##############################
# Calling Database Functions #
##############################
origin_json_results = process_origin_nexus_dataset(origin_nexus_processed)
nebula_json_results = process_nebula_nexus_dataset(nebula_nexus_processed)
horizon_json_results = process_horizon_nexus_dataset(horizon_nexus_processed)

generate_json_results(origin_json_results, 'origin')
generate_json_results(nebula_json_results, 'nebula')
generate_json_results(horizon_json_results, 'horizon')

#################################
# Image Processing & Displaying #
#################################
# Convert the processed image back to BGR for displaying with OpenCV.
image_bgr = cv2.cvtColor(image_mediapipe, cv2.COLOR_RGB2BGR)
# Display the original image with pose landmarks.
cv2.imshow('Output BGR', image_bgr)
# Display the blueprint grid.
cv2.imshow('Blueprint with Landmarks', blueprint_processed_grid)
cv2.waitKey(0)

cv2.destroyAllWindows()
