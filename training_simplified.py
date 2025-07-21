import os
import time 
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, Value
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


progress_var = Value('i', 0)

def computeDataOnBorder(q_init, N_guess, N_increment, vboc_repeat, box_min_values, box_max_values):
    global progress_var
    controller.resetHorizon(N_guess)

    # Set velocity direction
    if args['check']:
        d = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    else:
        d = np.array([random.uniform(-1, 1) for _ in range(model.nv)])

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

    with progress_var.get_lock():  # Ensure thread-safe access
        progress_var.value += 1  # Increment the count of completed tasks
        if progress_var.value % 100 == 0: 
            print(f" Progress: {progress_var.value} / {controller.model.params.prob_num}")

    if x_star is None:
        return None, None, None, box_min_values, box_max_values, status, d
    else:
        return x_star[0], x_star, u_star, box_min_values, box_max_values, status, d

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

    ### PARSE ARGUMENTS+
    global args
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

    params.plot = False

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
        'silu': torch.nn.SiLU(),
        'sigm': torch.nn.Sigmoid()
    }
    act_fun = nls[act]
    nn_filename = f'{params.NN_DIR}_{act}.pt'
    print(f'{params.NN_DIR}_{act}.pt')
    if act in ['tanh', 'sine']:
        # ub = max(model.x_max[nq:]) * np.sqrt(nq)    # NOTE: check this
        ub = 1
    else:
        ub = 1
    

    print(f'ub {ub}')
    vel_considered = 1

    # generate the data

    # generate 100000 vectors and train network to compute its norm just for training test

    dim_v = 5
    n_norm = 5
    x_data = 1000*np.random.normal(size=(20000, dim_v))

    
    # Remove positions and stack box dimension
    np.random.shuffle(x_data)

    nn_model = NeuralNetwork(dim_v, params.hidden_size, 1, params.hidden_layers, act_fun, ub).to(device)
    loss_fn = torch.nn.MSELoss()
    # loss_fn = CustomLoss()
    optimizer = torch.optim.Adam(nn_model.parameters(), 
                                    lr=params.learning_rate,
                                    #weight_decay=2e-5,
                                    amsgrad=True)
    regressor = RegressionNN(params, nn_model, loss_fn, optimizer)

    # Compute outputs and inputs                                   
    n = x_data.shape[0]
    # mean = np.mean(x_data[:, :nbori])
    # std = np.std(x_data[:, :nbori])
    # x_data[:, :nbori] = (x_data[:, :nbori] - mean) / std
    y_data = np.linalg.norm(x_data, axis=1).reshape(n,1)    # correct only velocities
    # for k in range(n):
    #     if y_data[k] != 0.: 
    #         x_data[k, nbori:nbori+vel_considered] /= y_data[k] 

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

    print('***MODEL EVALUATION***')
    rmse_train, rel_err = regressor.testing(x_train_val, y_train_val)
    print(f'RMSE on Training data: {rmse_train:.5f}')
    print(f'Maximum error wrt training data: {torch.max(rel_err).item():.5f}')
    x_test, y_test = x_data[-test_size:], y_data[-test_size:]
    rmse_test, rel_err = regressor.testing(x_test, y_test)
    print('---')
    print(f'RMSE on Test data: {rmse_test:.5f}')
    print(f'Mean and std of the relative error: {torch.mean(rel_err).item()*100:.2f}% +/- {torch.std(rel_err).item()*100:.2f}%')
    print(f'99 % of the data has a relative error lower than: {torch.quantile(rel_err, 0.99).item()*100:.2f}%')
    print(f'Maximum relative error wrt test data: {torch.max(rel_err).item()*100:.2f}%')
    print(f'Minimum relative error wrt test data: {torch.min(rel_err).item()*100:.2f}%')
    print('*---*---*---*\n')

    

    # Plot the loss evolution
    plt.figure()
    plt.grid(True, which='both')
    plt.semilogy(train_evol, label='Training', c='b', lw=2)
    plt.semilogy(val_evol, label='Validation', c='g', lw=2)
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss (LP filtered)')
    plt.title(f'Training evolution, horizon {N}')
    plt.savefig(params.DATA_DIR + f'evolution_{N}, norm of {n_norm} components of the vector.png')

    plt.show()

    # Save the model
    torch.save({'model': nn_model.state_dict()}, nn_filename)

    dim_vect_test = 1000
    vect_test = np.random.rand(dim_vect_test,dim_v) * 20
    vect_y = np.linalg.norm(vect_test,axis=1)
    vect_y_ = nn_model.forward(torch.Tensor(vect_test).to(device)).detach().numpy()

    random_idx = np.random.randint(0, dim_vect_test, 50)

    plt.figure()
    plt.grid(True, which='both')
    plt.plot(vect_y[random_idx], label='True value', marker='o', linestyle='', c='g')
    plt.plot(vect_y_[random_idx], label='Training', marker='x', linestyle='', c='r')
    plt.legend()
    plt.savefig(params.DATA_DIR + f'norm of {n_norm} components of the vector.png')

    plt.show()
    #print(f'Test of the net: x = {vect_test}, norm = {np.linalg.norm(vect_test)}, network output = {nn_model.forward(torch.Tensor(vect_test).to(device))}')




    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f'Elapsed time: {hours}:{minutes:2d}:{seconds:2d}')

    
if __name__ == '__main__':
    main()