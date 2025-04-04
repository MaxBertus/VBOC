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

def computeDataOnBorder(q, N_guess, box_min_values, box_max_values):
    controller.resetHorizon(N_guess)

    # Randomize the initial state
    d = np.array([random.uniform(-1, 1) for _ in range(model.nv)])

    # Set the initial guess
    x_guess = np.zeros((N_guess, model.nx))
    u_guess = np.zeros((N_guess, model.nu))
    x_guess[:, :nq] = np.full((N_guess, nq), q)

    d /= np.linalg.norm(d)
    controller.setGuess(x_guess, u_guess)

    # Solve the OCP
    x_star, u_star, _, status = controller.solveVBOC(q, d, box_min_values, box_max_values, N_guess, n=N_increment, repeat=3)
    if x_star is None:
        return None, None, None, status
    else:
        return x_star[0], x_star, u_star, status
    
def fixedVelocityDir(N_guess, n_pts=100):
    """ Compute data on section of the viability kernel"""
    sec_pts = []
    status_list = []
    controller.resetHorizon(N_guess)
    for i in range(nq):
        # print('#### DOF n %d ####' % i)
        q_grid = np.linspace(model.x_min[i], model.x_max[i], n_pts)
        q_grid = np.tile(q_grid, 2)
        x_sec = np.empty((0, model.nx)) * np.nan 
        status_vec = np.empty(n_pts * 2) * np.nan
        for j in range(n_pts * 2):
            q_try = (model.x_max[:nq] + model.x_min[:nq]) / 2
            q_try[i] = q_grid[j]
            x_try = np.hstack([q_try, np.zeros(nq)])
            
            # if not controller.checkCollision(x_try) and params.obs_flag:
            #     continue
            # x_init = np.vstack([x_init, x_try])
            x_guess = np.zeros((N_guess, model.nx))
            u_guess = np.zeros((N_guess, model.nu))
            x_guess[:, :nq] = np.full((N_guess, nq), q_try)

            d = np.zeros(model.nv)
            d[i] = 1 if j < n_pts else -1

            controller.setGuess(x_guess, u_guess)
            x_star, _, _, status = controller.solveVBOC(q_try, d, N_guess, n=N_increment, repeat=5)
            if status == 0:
                x_sec = np.vstack([x_sec, x_star[0]])
            # else: 
            #     print('Point number %d' % j)
            #     controller.ocp_solver.print_statistics()
            #     print(controller.ocp_solver.get_stats('residuals'))
            #     print('Check collision (false mean collision): ', controller.checkCollision(x_try))
            status_vec[j] = status
            # else:
            #     print(f'No solution found at dof {i}, step {j}, flag: {status}')
        sec_pts.append(x_sec)
        status_list.append(status_vec)
    return sec_pts, status_list

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

if __name__ == '__main__':
    
    start_time = time.time()

    ### PARSE ARGUMENTS OF COMMAND LINE 
    args = parse_args()
    # Define the available systems
    available_systems = ['sth']
    try:
        if args['system'] not in available_systems:
            raise NameError
    except NameError:
        print('\nSystem not available! Available: ', available_systems, '\n')
        exit()
    params = Parameters(args['system']) 
    params.build = args['build']
    act = args['activation']

    ### DEFINE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(params)
    controller = ViabilityController(model)
    nq = model.nq

    # Check if data and nn folders exist, if not create it
    if not os.path.exists(params.DATA_DIR):
        os.makedirs(params.DATA_DIR)
    if not os.path.exists(params.NN_DIR):
        os.makedirs(params.NN_DIR)

    N = 100
    N_increment = 1
    horizon = args['horizon']
    try:
        if horizon < 1:
            raise ValueError
    except ValueError:
        print('\nThe horizon must be greater than 0!\n')
        exit()
    if horizon < N:
        N = horizon
        N_increment = 0   

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
        ub = max(model.x_max[nq:]) * np.sqrt(nq)    # TODO: check this
    else:
        ub = 1

    # DATA GENERATION
    # Generate random initial configurations
    pos_init = np.zeros((params.prob_num, model.npos))
    # orient_init = np.zeros((params.prob_num, model.norient))

    roll = np.random.uniform(-model.phi, model.phi, size=(params.prob_num, 1))
    pitch = np.sqrt(model.phi**2 - roll**2) * np.random.choice([1, -1], size=(params.prob_num, 1))
    yaw = np.random.uniform(-np.pi, np.pi, size=(params.prob_num, 1))
    orient_init = np.hstack([roll, pitch, yaw])

    q_init = np.hstack([pos_init, orient_init])

    # Generate random box
    box_min_values = np.array([np.random.uniform(model.box_occupancy[:3], model.env_dimensions[:3]) for _ in range(params.prob_num)])

    box_max_values = np.array([np.random.uniform(model.box_occupancy[3:], model.env_dimensions[3:]) for _ in range(params.prob_num)])


    print('Start data generation')
    with Pool(params.cpu_num) as p:
        # inputs --> (initial random configuration, horizon)
        # res = p.starmap(computeDataOnBorder, [(q0, N) for q0 in q_init])

        res = p.starmap(computeDataOnBorder, [(q0, N, box_min, box_max) for q0, box_min, box_max in zip(q_init, box_min_values, box_max_values)]) # TODO: check this
    

    # RESTART FROM HERE

    x_data_temp, x_t, u_t, status = zip(*res)
    x_data = np.vstack([i for i in x_data_temp if i is not None])
    x_traj = np.asarray([i for i in x_t if i is not None])
    u_traj = np.asarray([i for i in u_t if i is not None])

    solved = len(x_data)
    print('Perc solved/numb of problems: %.2f' % (solved / params.prob_num * 100))
    print('Total number of points: %d' % len(x_data))
    np.save(f'{params.DATA_DIR}{nq}dof_vboc{obs_string}', x_data)
    np.save(f'{params.DATA_DIR}{nq}dof_trajx{obs_string}', x_traj)
    # np.save(params.DATA_DIR + str(nq) + 'dof_traju', u_traj)

    

    # Plot trajectory solution
    PLOTSSS = 0

    if PLOTSSS:
        colors = np.linspace(0, 1, horizon)
        t = np.linspace(0, horizon * params.dt, horizon)
        def computed_torque(x, u):
            tau = np.zeros((len(x), nq))
            for i in range(len(x)):
                tau[i] = kin_dyn.mass_matrix(model.H_b, x[i, :nq])[6:, 6:] @ u[i] + \
                            kin_dyn.bias_force(model.H_b, x[i, :nq], np.zeros(6), x[i, nq:])[6:]
            return tau
        # clear the plots directory
        for file in os.listdir(params.DATA_DIR + '/plots'):
            os.remove(params.DATA_DIR + '/plots/' + file)
        for k in range(len(x_traj)):
            fig, ax = plt.subplots(2, 2)
            ax = ax.reshape(-1)
            for i in range(nq):
                ax[i].grid(True)
                ax[i].scatter(x_traj[k][:, i], x_traj[k][:, nq + i], c=colors, cmap='coolwarm')
                ax[i].set_xlim([model.x_min[i], model.x_max[i]])
                ax[i].set_ylim([model.x_min[nq + i], model.x_max[nq + i]])
                ax[i].set_title(f'Joint {i + 1}')
                ax[i].set_xlabel(f'q_{i + 1}')
                ax[i].set_ylabel(f'dq_{i + 1}')
            plt.suptitle(f'Trajectory {k + 1}')
            plt.tight_layout()
            plt.savefig(params.DATA_DIR + f'/plots/traj_{k + 1}.png')
            plt.close(fig)

            fig, ax = plt.subplots(2, 2)
            ax = ax.reshape(-1)
            for i in range(nq):
                ax[i].grid(True)
                ax[i].plot(t, x_traj[k][:, nq + i], label=f'v_{i + 1}')
                ax[i].plot(t, u_traj[k][:, i], label=f'a_{i + 1}')
                ax[i].axhline(model.x_min[nq + i], color='b', linestyle='--')
                ax[i].axhline(model.x_max[nq + i], color='b', linestyle='--')
                ax[i].set_xlabel('Time [s]')
                ax[i].set_ylabel('Velocity [rad/s]')
                # ax[i].set_ylim([model.x_min[nq + i], model.x_max[nq + i]])
                ax[i].legend()
            plt.suptitle(f'Trajectory {k + 1}')
            plt.tight_layout()
            plt.savefig(params.DATA_DIR + f'/plots/vel_{k + 1}.png')
            plt.close(fig)

            fig, ax = plt.subplots(2, 2)
            ax = ax.reshape(-1)
            for i in range(nq):
                ax[i].grid(True)
                ax[i].plot(t, computed_torque(x_traj[k], u_traj[k])[:, i], label=f'tau_{i + 1}')
                ax[i].set_xlabel('Time [s]')
                ax[i].set_ylabel('Torque [Nm]')
                ax[i].set_ylim([model.tau_min[i], model.tau_max[i]])
                ax[i].legend()
            plt.suptitle(f'Trajectory {k + 1}')
            plt.tight_layout()
            plt.savefig(params.DATA_DIR + f'/plots/torque_{k + 1}.png')
            plt.close(fig)

    # histogram of status
    plt.figure()
    plt.hist(status, bins=[0, 1, 2, 3, 4, 5], edgecolor='black', align='left', rwidth=0.8)
    plt.title('Histogram of status flags')
    plt.xlabel('Flag')
    plt.ylabel('Frequency')
    plt.xticks(range(5))

    # TRAINING
    if args['training']:
        # Load the data
        x_data = np.load(f'{params.DATA_DIR}{nq}dof_vboc{obs_string}.npy')
        np.random.shuffle(x_data)
        
        nn_model = NeuralNetwork(model.nx, 256, 1, act_fun, ub).to(device)
        loss_fn = torch.nn.MSELoss()
        # loss_fn = CustomLoss()
        optimizer = torch.optim.Adam(nn_model.parameters(), 
                                     lr=params.learning_rate,
                                    #  weight_decay=2e-5,
                                     amsgrad=True)
        regressor = RegressionNN(params, nn_model, loss_fn, optimizer)

        # Compute outputs and inputs
        n = len(x_data)
        mean = np.mean(x_data[:, :nq])
        std = np.std(x_data[:, :nq])
        x_data[:, :nq] = (x_data[:, :nq] - mean) / std
        y_data = np.linalg.norm(x_data[:, nq:], axis=1).reshape(n, 1)
        for k in range(n):
            if y_data[k] != 0.: 
                x_data[k, nq:] /= y_data[k] 

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

        print('Evaluate the model')
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
    if args['plot']:
        nn_data = torch.load(nn_filename)
        nn_model = NeuralNetwork(model.nx, 256, 1, act_fun, ub)
        nn_model.load_state_dict(nn_data['model'])

        print('Generate fixed velocity direction points on a grid')
        x_fixed, x_status = fixedVelocityDir(N, n_pts=100)
        plot_brs(params, model, controller, nn_model, nn_data['mean'], nn_data['std'], x_fixed, x_status)
    plt.show()

    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f'Elapsed time: {hours}:{minutes:2d}:{seconds:2d}')
