import numpy as np
from casadi import MX, cos, sin, tan, Function, vertcat, horzcat, cross

nq = 6
x = MX.sym("x", nq * 2)
u = MX.sym("x", nq)

# Extract Euler angles
euler_angles = x[3:6] 
roll, pitch, yaw = euler_angles[0], euler_angles[1], euler_angles[2]

# Rotation matrix around x-axis using CasADi symbolic functions
R_z = vertcat(
    horzcat(cos(yaw), -sin(yaw), 0),
    horzcat(sin(yaw), cos(yaw), 0),
    horzcat(0, 0, 1))

# Define CasADi function
R = Function('R', [x], [R_z])

# Evaluate for a specific value of roll (e.g., roll = π/4)
roll_value = np.pi / 2  # 45 degrees
x_numeric = np.zeros(nq * 2)  # Create a zero vector for x
x_numeric[5] = roll_value  # Assign roll

# Compute the result
result = R(x_numeric)

# Print the evaluated matrix
print(result @ np.array([[1], [0], [0]]))  # Example of multiplying with a vector


alpha = np.pi / 6
cf = 1.0
ct = 1.0
r = 1.0
mass = 1.0
g = 1.0
J = np.eye(3) 

sin_a = np.sin(alpha)
cos_a = np.cos(alpha)

F = cf * np.array([
[0, np.sqrt(3)/2 * sin_a, -np.sqrt(3)/2 * sin_a, 0, np.sqrt(3)/2 * sin_a, -np.sqrt(3)/2 * sin_a],
[sin_a, -1/2 * sin_a, -1/2 * sin_a, sin_a, -1/2 * sin_a, -1/2 * sin_a],
[cos_a, cos_a, cos_a, cos_a, cos_a, cos_a]
])

M = ct * np.array([
    [0, np.sqrt(3)/2 * r * cos_a - np.sqrt(3)/2 * sin_a, np.sqrt(3)/2 * r * cos_a - np.sqrt(3)/2 * sin_a, 0, -np.sqrt(3)/2 * r * cos_a + np.sqrt(3)/2 * sin_a, -np.sqrt(3)/2 * r * cos_a + np.sqrt(3)/2 * sin_a],
    [-r * cos_a + sin_a, -1/2 * r * cos_a + 1/2 * sin_a, 1/2 * r * cos_a - 1/2 * sin_a, r * cos_a - sin_a, 1/2 * r * cos_a - 1/2 * sin_a, -1/2 * r * cos_a + 1/2 * sin_a],
    [r * sin_a + cos_a, -r * sin_a - cos_a, r * sin_a + cos_a, -r * sin_a - cos_a, r * sin_a + cos_a, -r * sin_a - cos_a]
])

# Control force and torque
fc = Function('fc', [x, u], [R(x) @ F @ u])
tc = Function('tc', [u], [M @ u])

print(fc(x_numeric, np.array([[1], [0], [0], [0], [0], [0]])))
print(tc(np.array([[1], [0], [0], [0], [0], [0]])))


Tinv_expr = vertcat(
    horzcat(1, sin(roll)*tan(pitch), cos(roll)*tan(pitch)),
    horzcat(0, cos(roll), -sin(roll)),
    horzcat(0, sin(roll)/cos(pitch), cos(roll)/cos(pitch)))

Tinv = Function('Tinv', [x], [Tinv_expr])

f_expl = vertcat(
    x[nq:nq+3],
    Tinv(x)@x[nq+3:],
    g*np.array([0, 0, 1]) + fc(x, u)/mass, 
    np.linalg.inv(J) @ (cross(x[nq+3:], J @ x[nq+3:])) + np.linalg.inv(J) @ tc(u)
)

print(f_expl)