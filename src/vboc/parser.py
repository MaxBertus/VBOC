import os
import yaml
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--system', type=str, default='sth',
                        help='Systems to test. Available: sth (Star-shaped Tilted Hexarotor)')
    parser.add_argument('-b', '--build', action='store_true',
                        help='Build the code of the embedded controller')
    parser.add_argument('--horizon', type=int, default=None,
                        help='Horizon of the optimal control problem')
    parser.add_argument('-t', '--training', action='store_true',
                        help='Train the neural network model that approximates the viability kernel')
    parser.add_argument('-p', '--plot', action='store_true',
                        help='Plot the approximated viability kernel')
    parser.add_argument('-e', '--epochs', type=int, default=1000,
                        help='Number of epochs for training the neural network')
    parser.add_argument('-a', '--activation', type=str, default='relu',
                        help='Activation function for the neural network')
    parser.add_argument('-c', '--check', action='store_true',
                        help='Sanity check: fixed d and box, and null orientation')
    parser.add_argument('-g', '--generation', action='store_true',
                    help='Generate the dataset for training the neural network')
    return vars(parser.parse_args())

class Parameters:
    def __init__(self, robot_name):
        self.robot_name = robot_name       
        self.PKG_DIR = os.path.dirname(os.path.abspath(__file__))
        self.ROOT_DIR = os.path.join(self.PKG_DIR, '../..')
        self.CONF_DIR = os.path.join(self.ROOT_DIR, 'config/')
        self.DATA_DIR = os.path.join(self.ROOT_DIR, 'data/')
        self.GEN_DIR = os.path.join(self.ROOT_DIR, 'generated/')
        self.NN_DIR = os.path.join(self.ROOT_DIR, 'nn_models/' + robot_name + '/')

        parameters = yaml.load(open(self.ROOT_DIR + '/config.yaml'), Loader=yaml.FullLoader)

        self.prob_num = int(parameters['prob_num'])
        self.cpu_num = int(parameters['cpu_num'])
        # self.build = False
        
        self.N = int(parameters['N'])
        self.N_increment = int(parameters['N_increment'])
        self.dt = float(parameters['dt'])
        #self.alpha = int(parameters['alpha'])

        self.solver_type = 'SQP'
        self.solver_mode = parameters['solver_mode']
        self.nlp_max_iter = int(parameters['nlp_max_iter'])
        self.qp_max_iter = int(parameters['qp_max_iter'])
        self.alpha_reduction = float(parameters['alpha_reduction'])
        self.alpha_min = float(parameters['alpha_min'])
        self.levenberg_marquardt = float(parameters['levenberg_marquardt'])

        self.state_tol = float(parameters['state_tol'])
        self.cost_tol = float(parameters['cost_tol'])
        self.vboc_repeat = int(parameters['vboc_repeat'])
        self.globalization = 'MERIT_BACKTRACKING'

        self.learning_rate = float(parameters['learning_rate'])
        self.batch_size = int(parameters['batch_size'])
        self.beta = float(parameters['beta'])
        self.train_ratio = float(parameters['train_ratio'])
        self.val_ratio = float(parameters['val_ratio'])
        self.hidden_size = int(parameters['hidden_size'])
        self.hidden_layers = int(parameters['hidden_layers'])

        self.mass = float(parameters['mass'])
        self.J = np.array(parameters['J'])
        self.l = float(parameters['l'])
        self.cf = float(parameters['cf'])
        self.ct = float(parameters['ct'])
        self.u_bar = float(parameters['max_w'])**2
        self.alpha_tilt = np.deg2rad((float(parameters['alpha_tilt'])))

        self.min_width = float(parameters['min_width'])
        self.min_length = float(parameters['min_length'])
        self.min_height = float(parameters['min_height'])

        self.max_width = float(parameters['max_width'])
        self.max_length = float(parameters['max_length'])
        self.max_height = float(parameters['max_height'])

        self.orient_g_rej = bool(parameters['orient_g_rej'])