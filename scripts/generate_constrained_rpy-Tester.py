import numpy as np
from scipy.spatial.transform import Rotation as R

def generate_constrained_rpy(a_deg, b_deg, n_samples):
    """
    Generates N uniformly distributed orientations satisfying Z-angle constraint.
    Uses rejection sampling based on random quaternions.

    Args:
        a_deg (float): Minimum angle between original Z and rotated Z (degrees).
        b_deg (float): Maximum angle between original Z and rotated Z (degrees).
        n_samples (int): Number of orientations to generate.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            roll (n,), pitch (n,), yaw (n,) arrays in radians.
    """

    # Input validation
    if not (
        isinstance(a_deg, (int, float)) and
        isinstance(b_deg, (int, float)) and
        isinstance(n_samples, int) and
        0 <= a_deg <= b_deg <= 180 and
        n_samples >= 0
    ):
        raise ValueError("Invalid input arguments. Check ranges (0<=a<=b<=180) and types.")

    if n_samples == 0:
        return np.array([]), np.array([]), np.array([])

    a_rad = np.deg2rad(a_deg)
    b_rad = np.deg2rad(b_deg)

    roll_list = []
    pitch_list = []
    yaw_list = []

    count = 0
    tries = 0
    max_tries = max(n_samples * 100, 10000)

    while count < n_samples and tries < max_tries:
        tries += 1

        # Generate random quaternion and convert to rotation matrix
        quat = R.random().as_quat()  # [x, y, z, w]
        rot = R.from_quat(quat).as_matrix()

        # Check Z-angle constraint
        cos_theta = np.clip(rot[2, 2], -1.0, 1.0)
        theta_rad = np.arccos(cos_theta)

        if a_rad <= theta_rad <= b_rad:
            # Convert to euler and split to roll, pitch, yaw
            eul = R.from_matrix(rot).as_euler('ZYX')  # [yaw, pitch, roll]
            yaw, pitch, roll = eul
            roll_list.append(roll)
            pitch_list.append(pitch)
            yaw_list.append(yaw)
            count += 1

    if count < n_samples:
        print(f"Warning: Maximum tries ({max_tries}) exceeded. Found {count}/{n_samples} samples.")

    return (
        np.array(roll_list),
        np.array(pitch_list),
        np.array(yaw_list)
    )


if __name__ == "__main__":
    # Example usage
    a = 30
    b = 60
    n = 100

    roll, pitch, yaw = generate_constrained_rpy(a, b, n)

    print("Roll (degree):", np.rad2deg(roll))
    print("Pitch (degree):", np.rad2deg(pitch))
    print("Yaw (degree):", np.rad2deg(yaw))

    print("Inclination (degrees):", np.rad2deg(np.sqrt(np.square(roll) + np.square(pitch))))