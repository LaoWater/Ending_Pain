# ----------------------------------------------------------------------------------------------
# Ending Pain
#
# This tool is an image-processing tool for deep biomechanical readings and fixing plan.
#
# Author :  Neo & N1ptic
# ----------------------------------------------------------------------------------------------
import cv2
import mediapipe as mp
import numpy as np
from metrics_classes import StationsMetrics, TracksMetrics


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

    return blueprint_image


def calculate_compressions(stations_metrics_instance, tracks_metrics_instance):
    # Define the relationships between tracks and their corresponding stations.
    track_to_station_mapping = {
        "Train Track 1 (Nose to Shoulder)": ["Shoulders"],
        "Train Track 2 (Shoulder to Hip)": ["Shoulders", "Hips"],
        "Train Track 3 (Hip to Knees)": ["Knees", "Hips"],
        "Train Track 4 (Knees to Foot)": ["Knees", "Feet"],
        "Train Track 5 (Shoulder to Elbow)": ["Shoulders"],
    }

    def get_offsets_and_sort(metrics_dict):
        # Prepare a list of (name, offset) tuples
        name_offset_pairs = [
            (key, float(value['Offset'].replace('%', '')) if isinstance(value['Offset'], str)
             else float(value['Offset'])) for key, value in metrics_dict.items()
        ]
        # Sort based on the absolute value of offsets, but keep the tuples intact
        sorted_pairs = sorted(name_offset_pairs, key=lambda pair: abs(pair[1]), reverse=True)
        # Format the sorted pairs for printing and retain numeric data for calculations
        sorted_info_for_print = [f"{pair[0]}: {pair[1]}" for pair in sorted_pairs]
        return sorted_pairs, sorted_info_for_print

    # Assuming tracks_metrics_instance.tracks and stations_metrics_instance.metrics are defined elsewhere
    tracks_sorted_pairs, tracks_sorted_info_for_print = get_offsets_and_sort(tracks_metrics_instance.tracks)
    stations_sorted_pairs, stations_sorted_info_for_print = get_offsets_and_sort(stations_metrics_instance.metrics)

    # 70% Tracks weigh-in, 30% Stations
    def calculate_weighted_values(tracks_pairs, stations_pairs, mapping):
        weighted_results = []
        for w_track, track_value in tracks_pairs:
            if w_track in mapping:
                station_names = mapping[w_track]
                station_values = [value for name, value in stations_pairs if name in station_names]

                # Calculate the average station value if multiple stations are related to a single track
                avg_station_value = sum(station_values) / len(station_values) if station_values else 0

                # Calculate weighted value and round it
                weighted_value = round(0.7 * track_value + 0.3 * avg_station_value, 2)
                weighted_results.append((w_track, weighted_value, abs(weighted_value)))

        # Sort by the absolute values stored as the third element in each tuple
        weighted_results.sort(key=lambda x: x[2], reverse=True)

        # Remove the absolute values from the results, keeping the original weighted values
        final_results = [(weighted_track, value) for weighted_track, value, _ in weighted_results]

        return final_results

    # Calculate and sort the AI calculated values
    ai_sorted_results = calculate_weighted_values(tracks_sorted_pairs, stations_sorted_pairs, track_to_station_mapping)

    # Explicitly handling numerical operations for compression sums using the pairs
    right_tracks_compression = sum([pair[1] for pair in tracks_sorted_pairs if pair[1] > 0])
    left_tracks_compression = sum([abs(pair[1]) for pair in tracks_sorted_pairs if pair[1] < 0])
    right_stations_compression = sum([pair[1] for pair in stations_sorted_pairs if pair[1] > 0])
    left_stations_compression = sum([abs(pair[1]) for pair in stations_sorted_pairs if pair[1] < 0])

    compressions = {
        'tracks_offsets_sorted_for_print': tracks_sorted_info_for_print,
        'stations_offsets_sorted_for_print': stations_sorted_info_for_print,
        'compressions': {'right_tracks_compression': round(right_tracks_compression, 2),
                         'left_tracks_compression': round(left_tracks_compression, 2),
                         'right_stations_compression': round(right_stations_compression, 2),
                         'left_stations_compression': round(left_stations_compression, 2),
                         },
        'Left': round((0.7 * left_tracks_compression + 0.3 * left_stations_compression), 2),
        'Right': round((0.7 * right_tracks_compression + 0.3 * right_stations_compression), 2),
        'AI_Sorted_Results': ai_sorted_results
    }

    return compressions


# To print the results in a readable format:
def print_calculated_compressions(compressions):
    print("\n---------------------------------------------------------\n"
          "*Dictionary\nSorted Tracks and Stations will make up ai_sorted_results with 70%-30% weigh\n"
          "\nTracks Offsets (sorted):")
    for track_info in compressions['tracks_offsets_sorted_for_print']:
        print(track_info)
    print("\nStations Offsets (sorted):")
    for station_info in compressions['stations_offsets_sorted_for_print']:
        print(station_info)
    print("\nCompressions Summary:")
    print("Right Tracks Compression:", compressions['compressions']['right_tracks_compression'])
    print("Left Tracks Compression:", compressions['compressions']['left_tracks_compression'])
    print("Right Stations Compression:", compressions['compressions']['right_stations_compression'])
    print("Left Stations Compression:", compressions['compressions']['left_stations_compression'])

    # Printing the calculated Left and Right values
    print("\n---------------------------------------------------------\n\nWeighted Compressions (tracks and "
          "stations combined):\naka Ending_Pain Lateral Lines Compression Scale\n")
    print("Left Line Compression:", compressions['Left'])
    print("Right Line Compression:", compressions['Right'])
    print("\n---------------------------------------------------------\nAI Sorted Results Analysis:\n"
          "*Treatment Plan, Tracks & Stations Priority*\n", compressions['AI_Sorted_Results'])


###################
###################
# Starting Script #
###################
###################

stations_metrics = StationsMetrics()
image_path = 'Photos/Anterior_view_3.JPG'

pose_data, image_mediapipe = process_image_mediapipe(image_path)
blueprint_processed_grid = image_blueprint()

# Convert the processed image back to BGR for displaying with OpenCV.
image_bgr = cv2.cvtColor(image_mediapipe, cv2.COLOR_RGB2BGR)

# Display the original image with pose landmarks.
cv2.imshow('Output BGR', image_bgr)
# Display the blueprint grid.
cv2.imshow('Blueprint with Landmarks', blueprint_processed_grid)

# Updating Tracks class once Stations is complete.
tracks_metrics = TracksMetrics(stations_metrics)

###################################
# Formatting and Printing Results #
###################################

print("\nStations Metrics: \n*Dictionary \nPositive Offset = Higher Left side Station (Point) ~= Right Side Compression"
      "\nNegative Offset = Lower Left Side Station ~= Left Side Compression\n")
for part, metrics in stations_metrics.metrics.items():
    # Print general Offset for all parts
    print(f'{part}: Offset = {metrics["Offset"]}')

    # Automate handling for 'Hands' and 'Feet' to include both 'Left Analysis' and 'Right Analysis' details
    if part in ['Hands', 'Feet']:
        for side in ['Left', 'Right']:
            analysis_key = f'{side} Analysis'
            if analysis_key in metrics:
                analysis_details = ', '.join([f'{key}: {value}' for key, value in metrics[analysis_key].items()])
                print(f'{part}: {side} Analysis: {analysis_details}')
print('\n', end='')  # Add an extra newline for separation between parts

print("\nTrack Metrics: \n*Dictionary \nPositive Offset = Right Lateral Line(Side) Compression"
      "\nNegative Offset = Left LL. (Side) Compression\n")
for track, details in stations_metrics.tracksMetrics.tracks.items():
    print(f"{track}: {details}")

# Calculating Compressions of types Stations and Tracks
ai_calculated_compressions = calculate_compressions(stations_metrics, stations_metrics.tracksMetrics)
print_calculated_compressions(ai_calculated_compressions)

cv2.waitKey(0)
cv2.destroyAllWindows()
