# geometry_utils.py
import math
from typing import Tuple, List

import numpy as np

Point = Tuple[float, float]
Triangle = Tuple[Point, Point, Point]


# Modified sigmoid function used for standardization of stations & tracks.
def modified_sigmoid(x, k=0.15):  # Adjust 'k' to change the steepness
    return 2 / (1 + np.exp(-k * x)) - 1


# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of the activation function for backpropagation
def sigmoid_derivative(x):
    return x * (1 - x)


def quadrilateral_area(coords: List[Point]) -> float:
    """Calculate the area of a quadrilateral using the Shoelace formula."""
    area = 0.0
    n = len(coords)
    for i in range(n):
        j = (i + 1) % n
        area += coords[i][0] * coords[j][1]
        area -= coords[j][0] * coords[i][1]
    return abs(area) / 2.0


def triangle_center_and_area(vertices: Triangle) -> Tuple[Point, float]:
    """Calculate the center and area of a triangle."""
    (x1, y1), (x2, y2), (x3, y3) = vertices
    x_center = (x1 + x2 + x3) / 3
    y_center = (y1 + y2 + y3) / 3
    area = abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)
    return (x_center, y_center), area


def calculate_distance(point1: Point, point2: Point) -> float:
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def calculate_percentage_difference(left_length, right_length):
    # Handle division by zero
    if right_length == 0:
        # Return a formatted string indicating no comparison can be made
        return None  # Adjust as needed, e.g., to "0.00 %" or another placeholder
    # Calculate the percentage difference and format it as a string with two decimal places followed by '%'
    offset = round(((left_length - right_length) / right_length) * 100, 2)
    return offset


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
