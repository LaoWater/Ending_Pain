# tracks_processing.py

# Calculate Nexus Functions
# Nexus = Functional combinations of Tracks & Stations outputting real world functionality and movement.
# We will use Tracks Length comparison to calculate sigmoid output, reflecting Nexus L-R compression.

def calculate_origin_nexus(alpha_proximal, alpha_inferior, foot_rotation):
    # Origin Nexus connects body to ground.
    # Composed of LL. Proximal + LL. Inferior
    # Weights
    # Proximal Lateral Line weight
    w_p = 0.58
    # Inferior Lateral Line weight
    w_i = 0.35
    # Foot/Hip Rotation weight
    w_f = 0.07

    # Calculate the weighted output, currently disregarded foot rotation for simpler analysis
    output = (w_p * alpha_proximal) + (w_i * alpha_inferior) + (w_f * foot_rotation * 0)
    return round(output, 2)


def calculate_nebula_nexus(alpha_oblique_sling, alpha_proximal, lambda_os):
    # Center of kinetic energy Generation & Absorption
    # Composed of Oblique Sling + Lateral Line Proximal processing

    # Weights
    # Oblique Sling
    w_os = 0.58
    # Proximal Line
    w_p = 0.32
    # Oblique Sling Lambda angle
    w_os_angle = 0.10

    # Calculate the weighted output, currently disregarded foot rotation for simpler analysis
    output = (w_os * alpha_oblique_sling) + (w_p * alpha_proximal) + (w_os_angle * lambda_os)
    return round(output, 2)


def calculate_horizon_nexus(alpha_superior, alpha_arm_line, lambda_shoulder_rotation):
    # Functioning of extremities
    # We will use Tracks Length comparison to calculate sigmoid output, reflecting Nexus L-R compression.

    # Weights
    # Lateral Line Superior
    w_s = 0.70
    # Arm Line
    w_arm_l = 0.15
    # Shoulder Rotation
    w_shoulder_rot = 0.15

    # Calculate the weighted output, currently disregarded foot rotation for simpler analysis
    output = (w_s * alpha_superior) + (w_arm_l * alpha_arm_line) + (w_shoulder_rot * lambda_shoulder_rotation)
    return round(output, 2)
