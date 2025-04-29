import os
import time 
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
from urdf_parser_py.urdf import URDF
import adam 
from adam.numpy import KinDynComputations  
import torch
from vboc.parser import Parameters, parse_args
from vboc.abstract import Model
from vboc.controller import ViabilityController
from vboc.learning import NeuralNetwork, RegressionNN, plot_brs
from scipy.spatial.transform import Rotation as Rot
import shutil
from mpl_toolkits.mplot3d import Axes3D
import warnings
from rich.traceback import install
install()

def computeDataOnBorder(q_init, N_guess, N_increment, vboc_repeat, box_min_values, box_max_values):
    controller.resetHorizon(N_guess)

    # Randomize the initial state
    d = np.array([random.uniform(-1, 1) for _ in range(model.nv)])
    # d =np.array([1.0 for _ in range(model.nv)]) # FIXME: just for sanity check
    # d = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # FIXME: just for sanity check

    # Set the initial guess
    x_guess = np.zeros((N_guess, model.nx))
    #u_guess = np.zeros((N_guess, model.nu))  # NOTE: without gravity compensation
    u_guess = np.linalg.pinv(model.R(np.hstack((q_init, np.zeros(model.nx-model.nq)))).full() @ model.F) @ np.array([0, 0, model.mass * model.g])
    u_guess = np.full((N_guess, model.nu), u_guess)

    x_guess[:, :model.nq] = np.full((N_guess, model.nq), q_init)

    d /= np.linalg.norm(d)
    controller.setGuess(x_guess, u_guess)

    # Solve the OCP
    x_star, u_star, _, status = controller.solveVBOC(q_init, d, box_min_values, box_max_values, N_guess, n=N_increment, repeat=vboc_repeat)
    if x_star is None:
        return None, None, None, box_min_values, box_max_values, status, 
    else:
        return x_star[0], x_star, u_star, box_min_values, box_max_values, status
    
# def fixedVelocityDir(N_guess, N_increment, n_pts=100 ):  
#     """ Compute data on section of the viability kernel"""
#     sec_pts = []
#     status_list = []
#     controller.resetHorizon(N_guess)
#     for i in range(model.nq):
#         # print('#### DOF n %d ####' % i)
#         q_grid = np.linspace(model.x_min[i], model.x_max[i], n_pts)
#         q_grid = np.tile(q_grid, 2)
#         x_sec = np.empty((0, model.nx)) * np.nan 
#         status_vec = np.empty(n_pts * 2) * np.nan
#         for j in range(n_pts * 2):
#             q_try = (model.x_max[:model.nq] + model.x_min[:model.nq]) / 2
#             q_try[i] = q_grid[j]
#             x_try = np.hstack([q_try, np.zeros(model.nq)])
            
#             # if not controller.checkCollision(x_try) and params.obs_flag:
#             #     continue
#             # x_init = np.vstack([x_init, x_try])
#             x_guess = np.zeros((N_guess, model.nx))
#             u_guess = np.zeros((N_guess, model.nu))
#             x_guess[:, :model.nq] = np.full((N_guess, model.nq), q_try)

#             d = np.zeros(model.nv)
#             d[i] = 1 if j < n_pts else -1

#             controller.setGuess(x_guess, u_guess)
#             x_star, _, _, status = controller.solveVBOC(q_try, d, N_guess, n=N_increment, repeat=5)
#             if status == 0:
#                 x_sec = np.vstack([x_sec, x_star[0]])
#             # else: 
#             #     print('Point number %d' % j)
#             #     controller.ocp_solver.print_statistics()
#             #     print(controller.ocp_solver.get_stats('residuals'))
#             #     print('Check collision (false mean collision): ', controller.checkCollision(x_try))
#             status_vec[j] = status
#             # else:
#             #     print(f'No solution found at dof {i}, step {j}, flag: {status}')
#         sec_pts.append(x_sec)
#         status_list.append(status_vec)
#     return sec_pts, status_list

def generate_constrained_rpy(min_inclination, max_inclination, n_samples):
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
        isinstance(min_inclination, (int, float)) and
        isinstance(max_inclination, (int, float)) and
        isinstance(n_samples, int) and
        0 <= min_inclination <= max_inclination <= 180 and
        n_samples >= 0
    ):
        raise ValueError("Invalid input arguments. Check ranges (0<=a<=b<=180) and types.")

    if n_samples == 0:
        return np.array([]), np.array([]), np.array([])

    roll_list = []
    pitch_list = []
    yaw_list = []

    count = 0
    tries = 0
    max_tries = max(n_samples * 1000, 10000)

    while count < n_samples and tries < max_tries:
        tries += 1

        # Generate random quaternion and convert to rotation matrix
        quat = Rot.random().as_quat()  # [x, y, z, w]
        rot = Rot.from_quat(quat).as_matrix()

        # Check Z-angle constraint
        cos_theta = np.clip(rot[2, 2], -1.0, 1.0)
        theta = np.arccos(cos_theta)

        if min_inclination <= theta <= max_inclination:
            # Convert to euler and split to roll, pitch, yaw
            eul = Rot.from_matrix(rot).as_euler('ZYX')  # [yaw, pitch, roll]
            yaw, pitch, roll = eul
            roll_list.append(roll)
            pitch_list.append(pitch)
            yaw_list.append(yaw)
            count += 1

    if count < n_samples:
        raise RuntimeError(f"Error: Maximum tries ({max_tries}) exceeded. Found {count}/{n_samples} samples.")

    return (
        np.array(roll_list),
        np.array(pitch_list),
        np.array(yaw_list)
    )

def set_axes_equal(ax):
    """Set equal aspect ratio for a 3D plot."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max(x_range, y_range, z_range)

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
    ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
    ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])

class Sine(torch.nn.Module):  
    def __init__(self, alpha=1.):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return torch.sin(self.alpha * x)

class OverMSELoss(torch.nn.Module): 
    """ Custom MSE loss that penalizes more overestimates """
    def __init__(self, alpha=1., beta=0.6):
        super(OverMSELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_pred, y_true):
        l2 = torch.mean((y_pred - y_true) ** 2)
        l2_over = torch.mean(torch.relu(y_pred - y_true) ** 2) 
        return self.alpha * l2 + self.beta * l2_over
    
class RAELoss(torch.nn.Module): 
    """ Relative Absolute Error loss """
    def __init__(self):
        super(RAELoss, self).__init__()

    def forward(self, y_pred, y_true):
        num = torch.sum(torch.abs(y_true - y_pred))
        den = torch.sum(torch.abs(y_true - torch.mean(y_true)))
        return num / den
    
class CustomLoss(torch.nn.Module):  
    """ Custom loss function (MSE + RE on overestimates) """
    def __init__(self, alpha=1., beta=0.6):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_pred, y_true):
        l2 = torch.mean((y_pred - y_true) ** 2)
        l1_over = torch.mean(torch.relu(y_pred - y_true))
        return self.alpha * l2 + self.beta * l1_over 
    
def main():
    start_time = time.time()

    ### PARSE ARGUMENTS
    args = parse_args()
    robotic_system = args['system']
    available_systems = ['sth']
    try:
        if robotic_system not in available_systems:
            raise NameError
    except NameError:
        print('\nSystem not available! Available: ', available_systems, '\n')
        exit()
    params = Parameters(robotic_system) 
    params.build = args['build']
    act = args['activation']

    ### MODEL AND CONTROLLER DEFINITION
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global model, controller
    model = Model(params)
    controller = ViabilityController(model)
    nq = model.nq
    nu = model.nu

    if not os.path.exists(params.DATA_DIR):
        os.makedirs(params.DATA_DIR)
    if not os.path.exists(params.NN_DIR):
        os.makedirs(params.NN_DIR)

    N = params.N
    N_increment = params.N_increment
    vboc_repeat = params.vboc_repeat
    horizon = args['horizon']

    if horizon is not None:
        try:
            if horizon < 1:
                raise ValueError
        except ValueError:
            print('\nThe horizon must be greater than 0!\n')
            exit()
        if horizon < N:
            N = horizon

    nls = {
        'relu': torch.nn.ReLU(),
        'elu': torch.nn.ELU(),
        'tanh': torch.nn.Tanh(),
        'sine': Sine(),
        'gelu': torch.nn.GELU(approximate='tanh'),
        'silu': torch.nn.SiLU()
    }
    act_fun = nls[act]
    nn_filename = f'{params.NN_DIR}_{act}.pt'
    if act in ['tanh', 'sine']:
        # ub = max(model.x_max[nq:]) * np.sqrt(nq)    # NOTE: check this
        ub = 1
    else:
        ub = 100

    # DATA GENERATION
    # Initial position
    pos_init = np.zeros((params.prob_num, model.npos))

    # Initial orientation 
    if(params.orient_g_rej):
        min_phi = 0.0
        max_phi = model.phi_hovering
    else:
        min_phi = model.phi_max
        max_phi = np.pi/2 

    roll, pitch, yaw = generate_constrained_rpy(min_phi, max_phi, params.prob_num)
    orient_init = np.column_stack([roll, pitch, yaw])
    # orient_init = np.zeros((params.prob_num, model.nori)) # NOTE: non-random orientation

    q_init = np.hstack([pos_init, orient_init])

    # Obstacles box
    box_min_values = np.array([np.random.uniform([0.0, 0.0, 0.0], model.env_dimensions[:3]) for _ in range(params.prob_num)])
    box_max_values = np.array([np.random.uniform([0.0, 0.0, 0.0], model.env_dimensions[3:]) for _ in range(params.prob_num)])

    # box_min_values = np.array([model.env_dimensions[:3] for _ in range(params.prob_num)])
    # box_max_values = np.array([model.env_dimensions[3:] for _ in range(params.prob_num)])

    print('Start data generation')
    with Pool(params.cpu_num) as p:
        # inputs --> (initial random configuration, horizon)
        # res = p.starmap(computeDataOnBorder, [(q0, N) for q0 in q_init])

        res = p.starmap(computeDataOnBorder, [(q0, N, N_increment, vboc_repeat, box_min, box_max) for q0, box_min, box_max in zip(q_init, box_min_values, box_max_values)])

    x_0, x_t, u_t, b_m, b_M, status = zip(*res)

    if all(item is None for item in x_0):
        warnings.warn('No solution found for any problem. Exiting the program.', RuntimeWarning)
        print(status)
        exit()

    x_data = np.vstack([i for i in x_0 if i is not None])
    # x_traj = np.asarray([i for i in x_t if i is not None])
    # u_traj = np.asarray([i for i in u_t if i is not None])
    # b_min = np.asarray([i for i in b_m if i is not None])
    # b_max = np.asarray([i for i in b_M if i is not None])

    x_traj = [i for i in x_t if i is not None]
    u_traj = [i for i in u_t if i is not None]
    b_min = list(b_m)
    b_max = list(b_M)

    b_min_succ = [b_m[i] for i in range(len(b_m)) if x_0[i] is not None]
    b_max_succ = [b_M[i] for i in range(len(b_M)) if x_0[i] is not None]

    b_combined = np.vstack([np.hstack((b_min_succ[i], b_max_succ[i])) for i in range(len(b_min_succ))])

    solved = len(x_data)
    print('Perc solved/numb of problems: %.2f' % (solved / params.prob_num * 100))
    print('Total number of points: %d' % len(x_data))

    np.save(f'{params.DATA_DIR}{robotic_system}_x_vboc', x_data)
    np.save(f'{params.DATA_DIR}{robotic_system}_b_vboc', b_combined)

    # PLOT THE SOLUTIONS
    if params.plot_solutions:

        # Labels and titles
        pose_title = ['x', 'y', 'z', '$\phi$', '\u03B8', '$\gamma$']
        extended_pose_title = ['Position', 'Orientation', 'Inclination']
        velocities_title = ['Linear velocity', 'Angular velocity']
        pose_label = ['x [m]', 'y [m]', 'z [m]', 'r [rad]', 'p [rad]', 'y [rad]']
        pose_legend = ['x', 'y', 'z', 'r', 'p', 'y']
        vel_label = ['v$_x$ [m/s]', 'v$_y$ [m/s]', 'v$_z$ [m/s]', '$\omega_x$ [rad/s]', '$\omega_y$ [rad/s]', '$\omega_z$ [rad/s]']
        vel_legend = ['v$_x$', 'v$_y$', 'v$_z$', '$\omega_x$', '$\omega_y$', '$\omega_z$']
        y_lab_pose = ['Pos. [m]', 'Orient. [rad]', 'Incl. [rad]']
        y_lab_vel = ['v [m/s]', '$\omega$ [rad/s]']

        # Clear the plots directory and create subfolders
        plots_dir = os.path.join(params.DATA_DIR, 'plots')
        traj_dir = os.path.join(plots_dir, 'trajectories')
        pose_dir = os.path.join(plots_dir, 'poses')
        velocity_dir = os.path.join(plots_dir, 'velocities')
        input_dir = os.path.join(plots_dir, 'inputs')
        threeD_dir = os.path.join(plots_dir, '3D')

        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        else:
            for file in os.listdir(plots_dir):
                file_path = os.path.join(plots_dir, file)
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directories
                else:
                    os.remove(file_path)  # Remove files

        # Create subfolders for each type of plot
        os.makedirs(traj_dir, exist_ok=True)
        os.makedirs(pose_dir, exist_ok=True)
        os.makedirs(velocity_dir, exist_ok=True)
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(threeD_dir, exist_ok=True)

        # Start plotting 
        for k in range(len(x_traj)):

            horizon_ = x_traj[k].shape[0]
            colors = np.linspace(0, 1, horizon_)
            t = np.linspace(0, horizon_ * params.dt, horizon_)

            traj_xlim_min = b_min[k].tolist() + [-max_phi, -max_phi, -np.pi]
            traj_xlim_max = b_max[k].tolist() + [max_phi, max_phi, np.pi]

            # Plot the trajectory
            fig, ax = plt.subplots(2, 3)
            ax = ax.reshape(-1)
            for i in range(nq):
                ax[i].grid(True, linewidth=0.5)
                ax[i].scatter(x_traj[k][:, i], x_traj[k][:, nq + i], c=colors, cmap='coolwarm', s=1)
                ax[i].set_xlim([traj_xlim_min[i], traj_xlim_max[i]])
                ax[i].set_xlabel(f'{pose_label[i]}')
                ax[i].set_ylabel(f'{vel_label[i]}')
            plt.suptitle(f'Trajectory {k + 1}')
            plt.tight_layout()
            plt.savefig(os.path.join(traj_dir, f'traj_{k + 1}.png'))
            plt.close(fig)

            # Plot pose
            fig, ax = plt.subplots(3, 1)
            ax = ax.reshape(-1)
            j = 0
            for i in range(nq):
                if i == model.npos:
                    j += 1
                ax[j].grid(True)
                ax[j].set_title(f'{extended_pose_title[j]}')
                line, = ax[j].plot(t, x_traj[k][:, i], label=f'{pose_label[i]}')
                # ax[j].axhline(traj_xlim_min[i], color=line.get_color(), linestyle='--', linewidth=0.8)
                # ax[j].axhline(traj_xlim_max[i], color=line.get_color(), linestyle='--', linewidth=0.8)
                ax[j].set_xlabel('Time [s]')
                ax[j].set_ylabel(y_lab_pose[j])
                ax[j].legend()
            j += 1
            ax[j].grid(True)
            ax[j].set_title(f'{extended_pose_title[j]}')
            line, = ax[j].plot(t, np.sqrt(np.square(x_traj[k][:, 3]) + np.square(x_traj[k][:, 4])), label=f'{pose_label[i]}')
            ax[j].axhline(min_phi, color=line.get_color(), linestyle='--', linewidth=0.8)
            ax[j].axhline(max_phi, color=line.get_color(), linestyle='--', linewidth=0.8)
            ax[j].axhline(model.phi_hovering_max, color='r', linestyle='--', linewidth=0.8)
            ax[j].set_xlabel('Time [s]')
            ax[j].set_ylabel(y_lab_pose[j])
            ax[j].legend()

            plt.suptitle(f'Trajectory {k + 1}')
            plt.tight_layout()
            plt.savefig(os.path.join(pose_dir, f'pose_{k + 1}.png'))
            plt.close(fig)

            # Plot velocities
            fig, ax = plt.subplots(2, 1)
            ax = ax.reshape(-1)
            j = 0
            for i in range(nq):
                if i == model.npos:
                    j += 1
                ax[j].grid(True)
                ax[j].set_title(f'{velocities_title[j]}')
                ax[j].plot(t, x_traj[k][:, nq + i], label=f'{vel_label[i]}')
                ax[j].set_xlabel('Time [s]')
                ax[j].set_ylabel(y_lab_vel[j])
                ax[j].legend()
            plt.suptitle(f'Trajectory {k + 1}')
            plt.tight_layout()
            plt.savefig(os.path.join(velocity_dir, f'vel_{k + 1}.png'))
            plt.close(fig)

            # Plot the input
            offset = 200;
            fig, ax = plt.subplots()
            for i in range(nu):
                ax.grid(True)
                ax.plot(t, u_traj[k][:, i], label=f'u_{i + 1}')
                ax.set_title('Inputs')
                ax.axhline(model.u_bar, color='r', linestyle='--', lw=0.8)
                ax.set_xlabel('Time [s]')
                ax.set_ylabel('$u^2$ [(Hz/s)$^2$]')
                ax.set_ylim([0.0 - offset, model.u_bar+offset])
                ax.legend()
            plt.suptitle(f'Trajectory {k + 1}')
            plt.tight_layout()
            plt.savefig(os.path.join(input_dir, f'input_{k + 1}.png'))
            plt.close(fig)

            # Plot 3D positions
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(x_traj[k][:, 0], x_traj[k][:, 1], x_traj[k][:, 2], c=colors, cmap='coolwarm', s=10)
            
            
            # Add reference frames every 10th point
            for i in range(0, len(x_traj[k]), 10):
                roll, pitch, yaw = x_traj[k][i,3:6]
                rotation_matrix = Rot.from_euler('xyz', [roll, pitch, yaw]).as_matrix()

                # Define the arrow directions (unit vectors in local frame)
                arrow_length = 0.01  # Length of the arrows
                x_arrow = rotation_matrix[:, 0] * arrow_length
                y_arrow = rotation_matrix[:, 1] * arrow_length
                z_arrow = rotation_matrix[:, 2] * arrow_length

                # Plot the arrows
                ax.quiver(
                    x_traj[k][i,0], x_traj[k][i,1], x_traj[k][i,2],  # Arrow origin
                    x_arrow[0], x_arrow[1], x_arrow[2], color='b', label='X-axis' if i == 0 else ""
                )
                ax.quiver(
                    x_traj[k][i,0], x_traj[k][i,1], x_traj[k][i,2],
                    y_arrow[0], y_arrow[1], y_arrow[2], color='r', label='Y-axis' if i == 0 else ""
                )
                ax.quiver(
                    x_traj[k][i,0], x_traj[k][i,1], x_traj[k][i,2],
                    z_arrow[0], z_arrow[1], z_arrow[2], color='g', label='Z-axis' if i == 0 else ""
                )
                        
            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
            ax.set_zlabel('Z [m]')
            # ax.set_xlim(traj_xlim_min[0], traj_xlim_max[0])
            # ax.set_ylim(traj_xlim_min[1], traj_xlim_max[1])
            # ax.set_zlim(traj_xlim_min[2], traj_xlim_max[2])
            ax.set_title(f'3D Position Trajectory {k + 1}')
            set_axes_equal(ax)

            # Add a colorbar
            # plt.colorbar(sc, ax=ax, label='Time progression')

            # Save the figure
            plt.tight_layout()
            plt.savefig(os.path.join(threeD_dir, f'3D_traj_{k + 1}.png'))
            plt.close(fig)

            # Print the explicit dynamics for the last step
            # print(f'Explicit dynamics of trajectory {k + 1}:\n {model.f_expl_func(x_traj[k][-1, :], u_traj[k][-1, :]).full()}')
    
    # histogram of status
    # plt.figure()
    # plt.hist(status, bins=[0, 1, 2, 3, 4, 5], edgecolor='black', align='left', rwidth=0.8)
    # plt.title('Histogram of status flags')
    # plt.xlabel('Flag')
    # plt.ylabel('Frequency')
    # plt.xticks(range(5))
    # plt.show(block=False)

    # TRAINING
    if args['training']: # NOTE: to verify
        # Load the data
        x_data = np.load(f'{params.DATA_DIR}{robotic_system}_x_vboc.npy')
        b_data = np.load(f'{params.DATA_DIR}{robotic_system}_b_vboc.npy')

        # permutation = np.random.permutation(len(x_data))
        # x_data = x_data[permutation]
        # b_data = b_data[permutation]
        
        # Remove positions and stack box dimension
        x_data = np.hstack((b_data, x_data[:, 3:]))
        np.random.shuffle(x_data)

        nb = b_data.shape[1]
        nbori = nb+model.nori
        nx_train = nbori+model.nv

        nn_model = NeuralNetwork(nx_train, 256, 1, act_fun, ub).to(device)
        loss_fn = torch.nn.MSELoss()
        # loss_fn = CustomLoss()
        optimizer = torch.optim.Adam(nn_model.parameters(), 
                                     lr=params.learning_rate,
                                    #  weight_decay=2e-5,
                                     amsgrad=True)
        regressor = RegressionNN(params, nn_model, loss_fn, optimizer)

        # Compute outputs and inputs
        n = len(x_data)
        mean = np.mean(x_data[:, :nbori])
        std = np.std(x_data[:, :nbori])
        x_data[:, :nbori] = (x_data[:, :nbori] - mean) / std
        y_data = np.linalg.norm(x_data[:, nbori:], axis=1).reshape(n, 1) 
        for k in range(n):
            if y_data[k] != 0.: 
                x_data[k, nbori:] /= y_data[k] 

        train_size = int(params.train_ratio * n)
        val_size = int(params.val_ratio * n)
        test_size = n - train_size - val_size
        
        x_data = torch.Tensor(x_data).to(device)
        y_data = torch.Tensor(y_data).to(device)
        x_train_val, y_train_val = x_data[:-test_size], y_data[:-test_size]
        print('Start training\n')
        train_evol, val_evol = regressor.training(x_train_val, y_train_val, 
                                                  train_size, args['epochs'], refine=False)
        print('Training completed\n')

        print('MODEL EVALUATION')
        rmse_train, rel_err = regressor.testing(x_train_val, y_train_val)
        print(f'RMSE on Training data: {rmse_train:.5f}')
        print(f'Maximum error wrt training data: {torch.max(rel_err).item():.5f}')

        x_test, y_test = x_data[-test_size:], y_data[-test_size:]
        rmse_test, rel_err = regressor.testing(x_test, y_test)
        print(f'RMSE on Test data: {rmse_test:.5f}')
        print(f'99 % of the data has an error lower than: {torch.quantile(rel_err, 0.99).item():.5f}')
        print(f'Maximum error wrt test data: {torch.max(rel_err).item():.5f}')

        # Save the model
        torch.save({'mean': mean, 'std': std, 'model': nn_model.state_dict()}, nn_filename)

        # Plot the loss evolution
        plt.figure()
        plt.grid(True, which='both')
        plt.semilogy(train_evol, label='Training', c='b', lw=2)
        plt.semilogy(val_evol, label='Validation', c='g', lw=2)
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss (LP filtered)')
        plt.title(f'Training evolution, horizon {N}')
        plt.savefig(params.DATA_DIR + f'evolution_{N}.png')

        # # Plot the relative (mean) error evolution
        # plt.figure()
        # plt.grid(True, which='both')
        # plt.semilogy(err_evol, label='Rel Error', c='b', lw=2)
        # plt.legend()
        # plt.xlabel('Epochs')
        # plt.ylabel('Relative Error')
        # plt.title(f'Relative error evolution, horizon {N}')
        # plt.savefig(params.DATA_DIR + f'error_{N}.png')

        # # Box plot of the relative error
        # plt.figure()
        # plt.boxplot(rel_err.cpu().numpy(), notch=True)
        # plt.title('Box plot of the relative error')
        # plt.xlabel('Test data')
        # plt.ylabel('Relative error')
        # plt.savefig(params.DATA_DIR + 'boxplot.png')

        # # Difference between predicted and true values
        # with torch.no_grad():
        #     nn_model.eval()
        #     y_pred = nn_model(x_test).cpu().numpy()
        # plt.figure()
        # plt.plot(y_pred - y_test.cpu().numpy())
        # plt.xlabel('Test data')
        # plt.ylabel('Output')
        # plt.savefig(params.DATA_DIR + 'difference.png')

    # PLOT THE VIABILITY KERNEL
    # if args['plot']: 
    #     nn_data = torch.load(nn_filename)
    #     nn_model = NeuralNetwork(model.nx, 256, 1, act_fun, ub)
    #     nn_model.load_state_dict(nn_data['model'])

    #     print('Generate fixed velocity direction points on a grid')
    #     x_fixed, x_status = fixedVelocityDir(N, n_pts=100)
    #     plot_brs(params, model, controller, nn_model, nn_data['mean'], nn_data['std'], x_fixed, x_status)
    #     plt.show()

    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f'Elapsed time: {hours}:{minutes:2d}:{seconds:2d}')

    
if __name__ == '__main__':
    main()