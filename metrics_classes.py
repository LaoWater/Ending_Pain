import math


class PoseMetrics:
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
        # Extend metrics to include hips, elbows, knees
        self.metrics = {
            'Shoulders': {'Left': (0, 0), 'Right': (0, 0), 'DevianceScale': 0.00, 'Distance': 0, 'Y_offset': 0},
            'Hips': {'Left': (0, 0), 'Right': (0, 0), 'DevianceScale': 0.00, 'Distance': 0, 'Y_offset': 0},
            'Elbows': {'Left': (0, 0), 'Right': (0, 0), 'DevianceScale': 0.00, 'Distance': 0, 'Y_offset': 0},
            'Knees': {'Left': (0, 0), 'Right': (0, 0), 'DevianceScale': 0.00, 'Distance': 0, 'Y_offset': 0},
        }

    def update_landmark(self, index, coords):
        if index in self.landmarks:
            self.landmarks[index]['coords'] = coords
            self.calculate_metrics()

    def calculate_metrics(self):
        # Generalize metric calculation for multiple body parts
        for part in ['Shoulders', 'Hips', 'Elbows', 'Knees']:
            self._calculate_part_metrics(part)

    def _calculate_part_metrics(self, part):
        # Determine the indexes for left and right landmarks based on the part
        indexes = {
            'Shoulders': (11, 12),
            'Hips': (23, 24),
            'Elbows': (13, 14),
            'Knees': (25, 26)
        }

        left_idx, right_idx = indexes[part]
        left = self.landmarks[left_idx]['coords']
        right = self.landmarks[right_idx]['coords']

        self.metrics[part]['Left'] = left
        self.metrics[part]['Right'] = right
        dx = abs(int(right[0] - left[0]))
        dy = right[1] - left[1]

        distance = math.sqrt(dx ** 2 + dy ** 2)
        deviance = distance - dx  # Simplified deviance calculation

        # Update metrics for the part
        self.metrics[part]['Distance'] = distance
        self.metrics[part]['Y_offset'] = dy
        self.metrics[part]['DevianceScale'] = deviance
