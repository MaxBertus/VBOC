import random
from xml.parsers.expat import model
import numpy as np
import torch
import torch.nn as nn
torch.set_printoptions(profile="full")
import matplotlib.pyplot as plt
import matplotlib.patches as patches    
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
from tqdm import tqdm
from urdf_parser_py.urdf import URDF
import adam
from adam.pytorch import KinDynComputations

class NeuralNetwork(nn.Module):
    """ A simple feedforward neural network. """
    def __init__(self, input_size, hidden_size, output_size, number_hidden, activation=nn.ReLU(), ub=None):
        super().__init__()
        layers=[]

        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(activation)
        
        # Hidden layers
        for _ in range(number_hidden):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation)
        
        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))
        layers.append(activation)

        self.linear_stack = nn.Sequential(*layers)

        self.ub = ub if ub is not None else 1
        self.initialize_weights()

        #self.input_size = input_size

    def forward(self, x):
        #out = self.linear_stack(x[:,:self.input_size])* self.ub 
        out = self.linear_stack(x) * self.ub 

        return out #(out + 1) * self.ub / 2
    
    def initialize_weights(self):
        for layer in self.linear_stack:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)


class Sine(torch.nn.Module):
    def __init__(self, alpha=1.):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return torch.sin(self.alpha * x)
    

class NovelNeuralNetwork(nn.Module):
    """ MLP with distance function at the output layer. """
    def __init__(self, params, activation='relu', v_max=None):
        super().__init__()
        
        input_size = params.nx
        hidden_size = params.hidden_size
        hidden_layers = params.hidden_layers

        nls = {'relu': nn.ReLU(),
               'elu': nn.ELU(),
               'tanh': nn.Tanh(),
               'sine': Sine()}
        
        if activation not in nls.keys():
            raise ValueError(f'Activation function {activation} not implemented')

        nl = nls[activation]
        net = [nn.Linear(input_size, hidden_size), nl]
        for _ in range(hidden_layers):
            net.append(nn.Linear(hidden_size, hidden_size))
            net.append(nls[activation])
        net.append(nn.Linear(hidden_size, 1))
        net.append(nls[activation])

        print(*net)
        self.model = nn.Sequential(*net)

        self.v_max = v_max if v_max is not None else 1

    def forward(self, x):
        return self.model(x) * self.v_max 


class RegressionNN:
    """ Class that compute training and test of a neural network. """
    def __init__(self, params, model, loss_fn, optimizer):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.beta = params.beta
        self.batch_size = params.batch_size
        self.plot_train = params.plot
        self.plot_dir = params.PLOTS_DIR

    def training(self, x_train, y_train, x_val, y_val, epochs):
        """ Training of the neural network. """

        progress_bar = tqdm(total=epochs, desc='Training')

        loss_evol_train = []
        loss_evol_val = []
        loss_lp = 1
        plot_epochs = 500
        # plot_epochs = epochs / plot_epochs
        n = len(x_train)
        for ep in range(epochs):
            self.model.train()
            # Shuffle the data
            idx = torch.randperm(n)
            x_perm, y_perm = x_train[idx], y_train[idx]
            # Split in batches 
            x_batches = torch.split(x_perm, self.batch_size)
            y_batches = torch.split(y_perm, self.batch_size)
            for x, y in zip(x_batches, y_batches):
                # Forward pass
                y_pred = self.model(x)

                # NAN CHECK
                if torch.isnan(y_pred).any():
                    print(f"NaN detected in outputs at epoch {ep}")
                    break
                # Compute the loss
                loss = self.loss_fn(y_pred, y)

                if torch.isnan(loss).any():
                    print(f"NaN detected in loss at epoch {ep}")
                    break
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()

                for name, param in self.model.named_parameters():
                    if torch.isnan(param.grad).any():
                        print(f"NaN detected in gradients of {name} at epoch {ep}")
                        break

                self.optimizer.step()

                loss_lp = self.beta * loss_lp + (1 - self.beta) * loss.item()

            loss_evol_train.append(loss_lp)
            # Validation
            loss_val = self.validation(x_val, y_val)
            # if ep % 100 == 0: 
            #     print(f'Loss training: {loss_lp}')
            #     print(f'Loss validation: {loss_val}')
            loss_evol_val.append(loss_val)
            progress_bar.update(1)

            random_idx = np.random.randint(0, x_val.shape[0], 50)
            if ep % plot_epochs == 0 and ep > 0:
                self.plot_input_output(x_train[random_idx], x_val[random_idx], y_train[random_idx], y_val[random_idx],ep)

        progress_bar.close()
        return loss_evol_train, loss_evol_val

    def validation(self, x_val, y_val):
        """ Compute the loss wrt to validation data. """
        x_batches = torch.split(x_val, self.batch_size)
        y_batches = torch.split(y_val, self.batch_size)
        self.model.eval()
        tot_loss = 0
        y_out = []
        with torch.no_grad():
            for x, y in zip(x_batches, y_batches):
                y_pred = self.model(x)
                y_out.append(y_pred)
                loss = self.loss_fn(y_pred, y)
                tot_loss += loss.item()
            y_out = torch.cat(y_out, dim=0)
        return tot_loss / len(x_batches)
    
    def testing(self, x_test, y_test):
        """ Compute the RMSE wrt to training or test data. """
        x_batches = torch.split(x_test, self.batch_size)
        y_batches = torch.split(y_test, self.batch_size)
        self.model.eval()
        y_pred = []
        with torch.no_grad():
            for x, y in zip(x_batches, y_batches):
                y_pred.append(self.model(x))
            y_pred = torch.cat(y_pred, dim=0)
            rmse = torch.sqrt(mse_loss(y_pred, y_test)).item()
            rel_err = torch.abs(y_pred - y_test) / (torch.abs(y_test)+1e-8)
        return rmse, rel_err  

    def plot_input_output(self, input_test, input_val, true_output_test, true_output_val,epoch):
        with torch.no_grad():
            input = torch.Tensor(input_test).to(self.device)
            net_output_test = self.model(input).cpu().numpy()

            input = torch.Tensor(input_val).to(self.device)
            net_output_val = self.model(input_val).cpu().numpy()

        # Convert true outputs to numpy if they're tensors
        if isinstance(true_output_test, torch.Tensor):
            true_output_test = true_output_test.cpu().numpy()
        if isinstance(true_output_val, torch.Tensor):
            true_output_val = true_output_val.cpu().numpy()

        fig = plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.grid(True, which='both')
        plt.plot(true_output_test, label='True value', marker='o', linestyle='', c='g')
        plt.plot(net_output_test, label='Network output', marker='x', linestyle='', c='r')
        plt.legend()
        plt.title(f'Predicition training data')

        plt.subplot(1, 2, 2)
        plt.grid(True, which='both')
        plt.plot(true_output_val, label='True value', marker='o', linestyle='', c='g')
        plt.plot(net_output_val, label='Network output', marker='x', linestyle='', c='r')
        plt.legend()
        plt.title(f'Prediction validation data')

        fig.suptitle(f'Epoch {epoch}', fontsize=16)

        plt.savefig(self.plot_dir + '/training_validation/'+ f'training_validation_{epoch}.png')
        plt.show(block=False)

def plot_brs(params, model, controller, nn_model, mean, std, dataset, status_pts, grid=1.2e-2):
    """ Plot the Backward Reachable Set. """
    npos = model.npos
    color_map = ['green', 'red', 'orange', 'blue', 'purple']

    with torch.no_grad():
        for i in range(npos):
            plt.figure()
            print (f'Plotting BRS for position {i + 1}...')

            q_grid = np.arange(model.env_dimensions[i]- model.drone_occupancy[i], model.env_dimensions[i+model.npos] - model.drone_occupancy[i+model.npos] +  grid, grid)
            v_grid = np.arange(model.v_min[i], model.v_max[i] + grid, grid)
            box_max = np.array([model.drone_occupancy[3], model.drone_occupancy[4], model.drone_occupancy[5]])
            box_max_grid = np.tile(box_max, (len(q_grid),1))
            box_min_grid = box_max_grid.copy()

            for j in range(len(q_grid)):
                box_max_grid[j,i] = min(model.env_dimensions[i+3], model.env_dimensions[i+3] - q_grid[j])
                box_min_grid[j,i] = -max(model.env_dimensions[i], model.env_dimensions[i] - q_grid[j])

            box_max_grid = np.tile(box_max_grid, (len(v_grid), 1))
            box_min_grid = np.tile(box_min_grid, (len(v_grid), 1))

            q, v = np.meshgrid(q_grid, v_grid)
            q_rav, v_rav = q.ravel(), v.ravel()
            n = len(q_rav)

            nbori = 2*model.npos + model.nori
            x_static = np.zeros(nbori + model.nv)
            x = np.repeat(x_static.reshape(1, len(x_static)), n, axis=0)
            x[:, :model.npos] = box_min_grid
            x[:, model.npos:model.npos+3] = box_max_grid
            x[:, nbori + i] = v_rav

            # for k in range(len(x)):
            #     if k % 100 == 0:
            #         print(f'x[{k}]: {x[k]}')

            # Transform z-velocity
            # skew_col_idx = 8
            # x[:, skew_col_idx] = power_transfomer.transform(x[:, skew_col_idx].reshape(-1, 1)).ravel()
            
            x_in = np.copy(x)
            # Standardize box dimensions and initial orientation
            x_in[:, :nbori] = (x[:, :nbori] - mean) / std
            # Normalize velocities            
            y = np.linalg.norm(x[:, nbori:], axis=1)
            y = y.reshape(len(y),1)

            for k in range(len(y)):
                if y[k] != 0.: 
                    x_in[k, nbori:] /= y[k] 
            
            # Predict
            device = next(nn_model.parameters()).device  # get model device
            y_pred = nn_model(torch.from_numpy(x_in.astype(np.float32)).to(device)).cpu().numpy()
            out = np.array([0 if y[j] > y_pred[j] else 1 for j in range(n)])
            z = out.reshape(q.shape)
            plt.contourf(q, v, z, cmap='coolwarm', alpha=0.8)

            # Plot of the viable samples
            plt.scatter(dataset[i][:, i], dataset[i][:, model.nq + i], color='darkgreen', s=12)

            plt.xlim([model.env_dimensions[i]- model.drone_occupancy[i], model.env_dimensions[i+3]- model.drone_occupancy[i+model.npos]])
            plt.ylim([model.v_min[i], model.v_max[i]])
            plt.xlabel('pos_' + str(i + 1))
            plt.ylabel('vel_' + str(i + 1))
            plt.grid()
            plt.title(f"Classifier section position {i + 1}, horizon {controller.N}")
            plt.savefig(params.PLOTS_DIR + '/brs/' + f'{i + 1}_pos_{controller.N}_BRS.png')
            plt.show(block=False)