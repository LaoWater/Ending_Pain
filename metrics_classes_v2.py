from geometry_utils import (quadrilateral_area, calculate_distance, calculate_angle_with_vertical,
                            find_triangle_center_and_area, calculate_percentage_difference, modified_sigmoid)
import math


class StationsMetrics:
    def __init__(self):
        self.landmarks = {
            0: {'name': 'Nose', 'coords': (0, 0)},
            9: {'name': 'Mouth left', 'coords': (0, 0)},
            10: {'name': 'Mouth right', 'coords': (0, 0)},
            11: {'name': 'Left shoulder', 'coords': (0, 0)},
            12: {'name': 'Right shoulder', 'coords': (0, 0)},
            13: {'name': 'Left elbow', 'coords': (0, 0)},
            14: {'name': 'Right elbow', 'coords': (0, 0)},
            15: {'name': 'Left wrist', 'coords': (0, 0)},
            16: {'name': 'Right wrist', 'coords': (0, 0)},
            17: {'name': 'Left pinky', 'coords': (0, 0)},
            18: {'name': 'Right pinky', 'coords': (0, 0)},
            19: {'name': 'Left index', 'coords': (0, 0)},
            20: {'name': 'Right index', 'coords': (0, 0)},
            21: {'name': 'Left thumb', 'coords': (0, 0)},
            22: {'name': 'Right thumb', 'coords': (0, 0)},
            23: {'name': 'Left hip', 'coords': (0, 0)},
            24: {'name': 'Right hip', 'coords': (0, 0)},
            25: {'name': 'Left knee', 'coords': (0, 0)},
            26: {'name': 'Right knee', 'coords': (0, 0)},
            27: {'name': 'Left ankle', 'coords': (0, 0)},
            28: {'name': 'Right ankle', 'coords': (0, 0)},
            29: {'name': 'Left heel', 'coords': (0, 0)},
            30: {'name': 'Right heel', 'coords': (0, 0)},
            31: {'name': 'Left foot index', 'coords': (0, 0)},
            32: {'name': 'Right foot index', 'coords': (0, 0)}
        }

        self.metrics = {
            'Shoulders': {'Alpha': 0},
            'Hips': {'Alpha': 0},
            'Elbows': {'Alpha': 0},
            'Knees': {'Alpha': 0},
            'Hands': {'Left Analysis': {'Rotation': 'Neutral', 'Rotation Degree': 0},
                      'Right Analysis': {'Rotation': 'Neutral', 'Rotation Degree': 0},
                      'Alpha': 0},
            'Feet': {'Left Analysis': {'Center': (0, 0), 'Rotation': 'Neutral', 'Rotation Degree': 0},
                     'Right Analysis': {'Center': (0, 0), 'Rotation': 'Neutral', 'Rotation Degree': 0},
                     'Alpha': 0},
        }

        # Extend metrics to include tracks, initializing TracksMetrics
        self.tracksMetrics = TracksMetrics(self)

    def update_landmark(self, index, coords):
        if index in self.landmarks:
            self.landmarks[index]['coords'] = coords
        self.calculate_metrics()

    def calculate_tracks(self):
        self.tracksMetrics.calculate_tracks()  # Trigger tracks calculation

    def calculate_metrics(self):
        for part in ['Shoulders', 'Hips', 'Elbows', 'Knees']:
            self._calculate_stations_metrics(part)
        self._calculate_hand_rotation()
        self._calculate_foot_metrics()

    def _calculate_stations_metrics(self, part):
        # Metric calculation for shoulders, hips, elbows, knees
        indexes = {
            'Shoulders': (11, 12),
            'Hips': (23, 24),
            'Elbows': (13, 14),
            'Knees': (25, 26)
        }
        left_idx, right_idx = indexes[part]
        left = self.landmarks[left_idx]['coords']
        right = self.landmarks[right_idx]['coords']
        dy = left[1] - right[1]
        # distance = math.sqrt(dx ** 2 + dy ** 2)
        self.metrics[part]['Alpha'] = dy

    def _calculate_hand_rotation(self):
        left_hand_center, right_hand_center = None, None
        # Calculate rotation for hands
        for hand in ['Left Analysis', 'Right Analysis']:
            if hand == 'Left Analysis':
                wrist_idx, pinky_idx, thumb_idx = 15, 17, 21
            else:
                wrist_idx, pinky_idx, thumb_idx = 16, 18, 22
            wrist = self.landmarks[wrist_idx]['coords']
            pinky = self.landmarks[pinky_idx]['coords']
            thumb = self.landmarks[thumb_idx]['coords']

            # Forming quadrilateral vertices: wrist, thumb, pinky, and again wrist to close the shape
            quadrilateral = [wrist, thumb, pinky, wrist]
            area = quadrilateral_area(quadrilateral)

            if hand == 'Left Analysis':
                left_hand_center = find_triangle_center_and_area(wrist, pinky, thumb)
                rotation = 'Internal' if pinky[0] > thumb[0] else 'External'
            else:
                right_hand_center = find_triangle_center_and_area(wrist, pinky, thumb)
                rotation = 'Internal' if pinky[0] < thumb[0] else 'External'

            # Calculate sigmoid to transform the normalized value to a range from 0 to 1
            # sigmoid_output = round(1 / (1 + math.exp(-area)), 2)
            self.metrics['Hands'][hand]['Rotation'] = rotation
            self.metrics['Hands'][hand]['Rotation Degree'] = area

        formatted_Alpha = round(left_hand_center[1] - right_hand_center[1], 2)
        self.metrics['Hands']['Alpha'] = formatted_Alpha

    def _calculate_foot_metrics(self):
        left_foot_center, right_foot_center = None, None
        for foot in ['Left Analysis', 'Right Analysis']:
            if foot == 'Left Analysis':
                ankle_idx, heel_idx, foot_index_idx = 27, 29, 31
            else:
                ankle_idx, heel_idx, foot_index_idx = 28, 30, 32

            ankle = self.landmarks[ankle_idx]['coords']
            heel = self.landmarks[heel_idx]['coords']
            foot_index = self.landmarks[foot_index_idx]['coords']

            # Assuming find_triangle_center and _quadrilateral_area or similar are defined elsewhere
            triangle_data = find_triangle_center_and_area(ankle, heel, foot_index)
            # Applying Normalization
            area = triangle_data[2] / 200

            if foot == 'Left Analysis':
                left_foot_center = find_triangle_center_and_area(ankle, heel, foot_index)
                self.metrics['Feet'][foot]['Center'] = left_foot_center
                rotation = 'External' if foot_index[0] > heel[0] else 'Internal'
            else:
                right_foot_center = find_triangle_center_and_area(ankle, heel, foot_index)
                self.metrics['Feet'][foot]['Center'] = right_foot_center
                rotation = 'External' if foot_index[0] < heel[0] else 'Internal'

            sigmoid_area = round(1 / (1 + math.exp(-area)), 2)
            self.metrics['Feet'][foot]['Rotation'] = rotation
            self.metrics['Feet'][foot]['Rotation Degree'] = sigmoid_area

        formatted_Alpha = round(left_foot_center[1] - right_foot_center[1], 2)
        self.metrics['Feet']['Alpha'] = formatted_Alpha


####################
# Tracks class #
# Alpha -  reflect compression #
# Lambda -  reflect weight distribution #
#################################################

class TracksMetrics:
    def __init__(self, pose_metrics_instance):
        self.pose_metrics_instance = pose_metrics_instance
        # Tracks structure initialized with placeholders for calculated values
        self.tracks = {

            'Lateral Line': {'Alpha': 0, 'Lambda': 0},  # Train Track 1, Foot to Shoulder
            'Oblique Sling': {'Alpha': 0, 'Lambda': 0},  # Train Track 2, Oblique Sling
            'Lateral Line Proximal': {'Alpha': 0, 'Lambda': 0},  # Train Track 3, Shoulder to Hip
            'Lateral Line Inferior': {'Alpha': 0, 'Lambda': 0},  # Train Track 4, Hip to Foot
            'Lateral Line Superior': {'Alpha': 0, 'Lambda': 0},  # Train Track 5, Nose to Shoulder
            'Arm Line': {'Alpha': 0, 'Lambda': 0},  # Train Track 6, Nose to Shoulder
        }
        # Track weights
        self.track_weights = {
            'Lateral Line': 23,
            'Oblique Sling': 17,
            'Lateral Line Proximal': 22,
            'Lateral Line Inferior': 33,
            'Lateral Line Superior': 5,
            'Arm Line': 0,
        }

    def calculate_tracks(self):
        print("Calculating tracks...")  # Debugging
        # Assuming landmarks are available as a dictionary in the pose_metrics_instance
        landmarks = self.pose_metrics_instance.landmarks
        if not landmarks:
            print("Landmarks are not available for calculation.")
            return

        # For Track 4, assuming find_triangle_center_and_area returns (center_x, center_y, area)
        left_foot_center = self.pose_metrics_instance.metrics['Feet']['Left Analysis']['Center']
        right_foot_center = self.pose_metrics_instance.metrics['Feet']['Right Analysis']['Center']

        self.calculate_and_update_track_with_foot(11, 12, left_foot_center, right_foot_center,
                                                  'Lateral Line')
        self.calculate_and_update_track(11, 12, 24, 23,
                                        'Oblique Sling')
        self.calculate_and_update_track(11, 12, 23, 24,
                                        'Lateral Line Proximal')
        self.calculate_and_update_track_with_foot(23, 24, left_foot_center, right_foot_center,
                                                  'Lateral Line Inferior')
        self.calculate_and_update_track(0, 0, 11, 12,
                                        'Lateral Line Superior')
        self.calculate_and_update_track(11, 12, 13, 14,
                                        'Arm Line')

        # Debug, After calculations, print or return self.tracks to see updated metrics
        print("Updated track metrics:", self.tracks)

    def calculate_and_update_track(self, left_reference, right_reference, left_idx, right_idx, track_name):
        # Calculate distances using landmarks from the pose_metrics_instance
        left_distance = calculate_distance(self.pose_metrics_instance.landmarks[left_reference]['coords'],
                                           self.pose_metrics_instance.landmarks[left_idx]['coords'])
        right_distance = calculate_distance(self.pose_metrics_instance.landmarks[right_reference]['coords'],
                                            self.pose_metrics_instance.landmarks[right_idx]['coords'])

        left_angle = (calculate_angle_with_vertical
                      (self.pose_metrics_instance.landmarks[left_reference]['coords'],
                       self.pose_metrics_instance.landmarks[left_idx]['coords']))
        right_angle = (calculate_angle_with_vertical
                       (self.pose_metrics_instance.landmarks[right_reference]['coords'],
                        self.pose_metrics_instance.landmarks[right_idx]['coords']))

        # Calculate abstract Lambda angle
        Lambda = round((left_angle - right_angle),2)
        # Adjust the track length related to the lesser angle with the angle Alpha percentage
        if left_angle > right_angle:
            left_distance *= (1 + Lambda / 100)
        else:
            right_distance *= (1 + Lambda / 100)

        # Debugging
        # print(f"Calculating {track_name}: Left Distance = {left_distance}, Right Distance = {right_distance}")
        # De-bugging
        print(f"\n{track_name} Left angle :", left_angle)
        print(f"\n{track_name} Right angle :", right_angle)
        # print("Train Track 2 angle Alpha:", angle_Alpha)

        Alpha = calculate_percentage_difference(left_distance, right_distance)

        # Apply Normalization & Standardization
        Alpha = round(modified_sigmoid(Alpha), 2)
        Lambda = round(modified_sigmoid(Lambda), 2)

        # Update the track information
        self.tracks[track_name]['Alpha'] = Alpha
        self.tracks[track_name]['Lambda'] = Lambda

    def calculate_and_update_track_with_foot(self, left_idx, right_idx, left_foot_center, right_foot_center,
                                             track_name):
        # Special handling for foot centers as they are calculated differently
        left_distance = calculate_distance(self.pose_metrics_instance.landmarks[left_idx]['coords'],
                                           left_foot_center)
        right_distance = calculate_distance(self.pose_metrics_instance.landmarks[right_idx]['coords'],
                                            right_foot_center)
        left_angle = (calculate_angle_with_vertical
                      (self.pose_metrics_instance.landmarks[left_idx]['coords'],
                       left_foot_center))
        right_angle = (calculate_angle_with_vertical
                       (self.pose_metrics_instance.landmarks[right_idx]['coords'],
                        right_foot_center))

        # Calculate abstract Alpha Track length difference
        Alpha = calculate_percentage_difference(left_distance, right_distance)

        # Calculate abstract Lambda angle
        Lambda = round(abs(left_angle - right_angle),2)

        print(f"\n{track_name} Left angle :", left_angle)
        print(f"\n{track_name} Right angle :", right_angle)

        # Apply Normalization & Standardization
        Alpha = round(modified_sigmoid(Alpha), 2)
        Lambda = round(modified_sigmoid(Lambda), 2)

        # Update the track information for foot
        self.tracks[track_name]['Alpha'] = Alpha
        self.tracks[track_name]['Lambda'] = Lambda

    def sum_of_lambdas(self):
        total_lambda_weighted = 0
        for track_name, track_data in self.tracks.items():
            lambda_value = float(track_data['Lambda'])
            track_weight = self.track_weights[track_name]

            if lambda_value > 7:
                lambda_value = 7 + (lambda_value - 7) * 0.5

            weighted_lambda = lambda_value * track_weight
            total_lambda_weighted += weighted_lambda

        # Normalization
        print("Sum of labmdas = ", total_lambda_weighted)
        print("(Before Normalization) \n")
        normalized_value = total_lambda_weighted / 100

        # Calculate sigmoid to transform the normalized value to a range from 0 to 1
        sigmoid_output = round( 1 / (1 + math.exp(-normalized_value)), 4)

        return sigmoid_output
