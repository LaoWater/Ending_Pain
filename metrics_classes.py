import math


def _quadrilateral_area(coords):
    # Calculate the area of a quadrilateral using the Shoelace formula
    n = len(coords)  # Should be 4 for a quadrilateral
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += coords[i][0] * coords[j][1]
        area -= coords[j][0] * coords[i][1]
    area = abs(area) / 2.0
    return area


def find_triangle_center_and_area(x, y, z):
    # Assuming points is a list of three tuples: [(x1, y1), (x2, y2), (x3, y3)]
    x1, y1 = x
    x2, y2 = y
    x3, y3 = z

    # Calculate the center of the triangle
    x_center = (x1 + x2 + x3) / 3
    y_center = (y1 + y2 + y3) / 3

    # Calculate the area of the triangle
    area = abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)

    return x_center, y_center, area


def calculate_distance(point1, point2):
    distance = math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    rounded_distance = round(distance, 2)
    return rounded_distance


def calculate_percentage_difference(left_length, right_length):
    # Handle division by zero
    if right_length == 0:
        # Return a formatted string indicating no comparison can be made
        return "N/A %"  # Adjust as needed, e.g., to "0.00 %" or another placeholder
    # Calculate the percentage difference and format it as a string with two decimal places followed by '%'
    offset = ((left_length - right_length) / right_length) * 100
    formatted_offset = "{:.2f} %".format(offset)
    return formatted_offset


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
            32: {'name': 'Right foot index', 'coords': (0, 0)},
        }

        # Extend metrics to include tracks, initializing TracksMetrics
        self.tracksMetrics = TracksMetrics(self)

        self.metrics = {
            'Shoulders': {'Left': (0, 0), 'Right': (0, 0), 'Offset': 0},
            'Hips': {'Left': (0, 0), 'Right': (0, 0), 'Offset': 0},
            'Elbows': {'Left': (0, 0), 'Right': (0, 0), 'Offset': 0},
            'Knees': {'Left': (0, 0), 'Right': (0, 0), 'Offset': 0},
            'Hands': {'Left': (0, 0), 'Right': (0, 0),
                      'Left Analysis': {'Rotation': 'Neutral', 'Rotation Degree': 0},
                      'Right Analysis': {'Rotation': 'Neutral', 'Rotation Degree': 0},
                      'Offset': 0},
            'Feet': {'Left': (0, 0), 'Right': (0, 0),
                     'Left Analysis': {'Rotation': 'Neutral', 'Rotation Degree': 0},
                     'Right Analysis': {'Rotation': 'Neutral', 'Rotation Degree': 0},
                     'Offset': 0},
        }

    def update_landmark(self, index, coords):
        if index in self.landmarks:
            self.landmarks[index]['coords'] = coords
        self.calculate_metrics()
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
        self.metrics[part]['Offset'] = dy
        self.metrics[part]['Left'] = left
        self.metrics[part]['Right'] = right

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
            area = _quadrilateral_area(quadrilateral)

            if hand == 'Left Analysis':
                left_hand_center = find_triangle_center_and_area(wrist, pinky, thumb)
                rotation = 'Internal' if pinky[0] > thumb[0] else 'External'
            else:
                right_hand_center = find_triangle_center_and_area(wrist, pinky, thumb)
                rotation = 'Internal' if pinky[0] < thumb[0] else 'External'

            self.metrics['Hands'][hand]['Rotation'] = rotation
            self.metrics['Hands'][hand]['Rotation Degree'] = area

        formatted_offset = round(left_hand_center[1] - right_hand_center[1], 2)
        self.metrics['Hands']['Offset'] = formatted_offset
        self.metrics['Hands']['Left'] = (int(left_hand_center[0]), int(left_hand_center[1]))
        self.metrics['Hands']['Right'] = (int(right_hand_center[0]), int(right_hand_center[1]))

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
            area = triangle_data[2]

            if foot == 'Left Analysis':
                left_foot_center = find_triangle_center_and_area(ankle, heel, foot_index)
                rotation = 'External' if foot_index[0] > heel[0] else 'Internal'
            else:
                right_foot_center = find_triangle_center_and_area(ankle, heel, foot_index)
                rotation = 'External' if foot_index[0] < heel[0] else 'Internal'

            self.metrics['Feet'][foot]['Rotation'] = rotation
            self.metrics['Feet'][foot]['Rotation Degree'] = area

        formatted_offset = round(left_foot_center[1] - right_foot_center[1], 2)
        self.metrics['Feet']['Offset'] = formatted_offset
        self.metrics['Feet']['Left'] = (int(left_foot_center[0]), int(left_foot_center[1]))
        self.metrics['Feet']['Right'] = (int(right_foot_center[0]), int(right_foot_center[1]))


class TracksMetrics:
    def __init__(self, pose_metrics_instance):
        self.pose_metrics_instance = pose_metrics_instance
        # Tracks structure initialized with placeholders for calculated values
        self.tracks = {
            'Train Track 1 (Nose to Shoulder)': {'Left Length': 0, 'Right Length': 0, 'Offset': 0},  # Nose to Shoulder
            'Train Track 2 (Shoulder to Hip)': {'Left Length': 0, 'Right Length': 0, 'Offset': 0},  # Shoulder to Hip
            'Train Track 3 (Hip to Knees)': {'Left Length': 0, 'Right Length': 0, 'Offset': 0},  # Hip to Knee
            'Train Track 4 (Knees to Foot)': {'Left Length': 0, 'Right Length': 0, 'Offset': 0},  # Knee to Foot
            'Train Track 5 (Shoulder to Elbow)': {'Left Length': 0, 'Right Length': 0, 'Offset': 0},  # S. to Elbow
        }

    def calculate_tracks(self):
        print("Calculating tracks...")  # Debugging
        # Assuming landmarks are available as a dictionary in the pose_metrics_instance
        landmarks = self.pose_metrics_instance.landmarks
        if not landmarks:
            print("Landmarks are not available for calculation.")
            return

        # Define the logic for each track using landmark indices
        # You should adjust the indices based on your landmarks structure
        # Nose - Calculated with mutual reference point
        self.calculate_and_update_track(0, 0, 11, 12,
                                        'Train Track 1 (Nose to Shoulder)')
        # Other Tracks
        self.calculate_and_update_track(11, 12, 23, 24,
                                        'Train Track 2 (Shoulder to Hip)')
        self.calculate_and_update_track(23, 24, 25, 26,
                                        'Train Track 3 (Hip to Knees)')

        # For Track 4, assuming find_triangle_center_and_area returns (center_x, center_y, area)
        left_foot_center = find_triangle_center_and_area(
            landmarks[27]['coords'], landmarks[29]['coords'], landmarks[31]['coords'])
        right_foot_center = find_triangle_center_and_area(
            landmarks[28]['coords'], landmarks[30]['coords'], landmarks[32]['coords'])
        self.calculate_and_update_track_for_foot(25, 26, left_foot_center, right_foot_center,
                                                 'Train Track 4 (Knees to Foot)')

        self.calculate_and_update_track(11, 12, 13, 14,
                                        'Train Track 5 (Shoulder to Elbow)')

        # Debug, After calculations, print or return self.tracks to see updated metrics
        print("Updated track metrics:", self.tracks)

    def calculate_and_update_track(self, left_reference, right_reference, left_idx, right_idx, track_name):
        # Calculate distances using landmarks from the pose_metrics_instance
        left_distance = calculate_distance(self.pose_metrics_instance.landmarks[left_reference]['coords'],
                                           self.pose_metrics_instance.landmarks[left_idx]['coords'])
        right_distance = calculate_distance(self.pose_metrics_instance.landmarks[right_reference]['coords'],
                                            self.pose_metrics_instance.landmarks[right_idx]['coords'])

        # Calculate Angle offset based on vertical line - for better Shoulder-Hip relevance
        if track_name == 'Train Track 2 (Shoulder to Hip)' and True:  # Specific logic for shoulder-to-hip track
            # Calculate angles with the vertical line
            left_angle = (self.calculate_angle_with_vertical
                          (self.pose_metrics_instance.landmarks[left_reference]['coords'],
                           self.pose_metrics_instance.landmarks[left_idx]['coords']))
            right_angle = (self.calculate_angle_with_vertical
                           (self.pose_metrics_instance.landmarks[right_reference]['coords'],
                            self.pose_metrics_instance.landmarks[right_idx]['coords']))

            # Calculate angle offset percentage
            angle_offset = abs(left_angle - right_angle) / 2

            # De-bugging
            print("Train Track 2 left angle calculated:", left_angle)
            print("Train Track 2 right angle calculated:", right_angle)
            print("Train Track 2 angle offset:", angle_offset)

            # Adjust the track related to the lesser angle with the angle offset percentage
            if left_angle > right_angle:
                left_distance *= (1 + angle_offset / 100)
            else:
                right_distance *= (1 + angle_offset / 100)

        # Debugging
        print(f"Calculating {track_name}: Left Distance = {left_distance}, Right Distance = {right_distance}")

        offset = calculate_percentage_difference(left_distance, right_distance)

        # Update the track information
        self.tracks[track_name]['Left Length'] = round(left_distance, 2)
        self.tracks[track_name]['Right Length'] = round(right_distance, 2)
        self.tracks[track_name]['Offset'] = offset

    def calculate_and_update_track_for_foot(self, left_idx, right_idx, left_foot_center, right_foot_center, track_name):
        # Special handling for foot centers as they are calculated differently
        left_distance = calculate_distance(self.pose_metrics_instance.landmarks[left_idx]['coords'],
                                           left_foot_center)
        right_distance = calculate_distance(self.pose_metrics_instance.landmarks[right_idx]['coords'],
                                            right_foot_center)
        offset = calculate_percentage_difference(left_distance, right_distance)

        # Update the track information for foot
        self.tracks[track_name]['Left Length'] = left_distance
        self.tracks[track_name]['Right Length'] = right_distance
        self.tracks[track_name]['Offset'] = offset

    @staticmethod
    def calculate_angle_with_vertical(point1, point2):
        dx = point2[0] - point1[0] + 1e-6  # Small term to prevent dx from being exactly zero
        dy = point2[1] - point1[1]

        # Check if dx is very small, indicating a nearly vertical line
        if abs(dx) < 1e-6:
            # If the line is almost vertical, it makes a 90-degree angle with the horizontal
            angle_deg = 90.0
        else:
            slope = dy / dx
            # Use atan to find the angle of the slope, then find its negative reciprocal
            # Since atan returns radians, convert to degrees
            if slope == 0:  # Directly handling the case where slope is exactly zero
                angle_rad = math.pi / 2  # Vertical angle
            else:
                angle_rad = math.atan(-1 / slope)  # Calculating angle with vertical
            angle_deg = math.degrees(angle_rad)

        return abs(angle_deg)
