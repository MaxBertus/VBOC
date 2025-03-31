import re
import numpy as np
from casadi import MX, vertcat, dot, Function
from urdf_parser_py.urdf import URDF
import adam
from adam.casadi import KinDynComputations
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

class Model:
    def __init__(self, params):
        self.params = params
        
        # System parameters
        self.mass = params.mass
        self.J = params.J
        self.l = params.l 
        self.cf = params.cf
        self.ct = params.ct
        self.r = self.cf / self.cf * self.l
        self.u_bar = params.u_bar
        self.alpha = params.alpha

        nq = 6 # dimension of pose: 3 for position, 3 for orientation (Euler Angles) 
        nu = 6 # dimension of input: 6 squared spinning rates

        self.amodel = AcadosModel()
        self.amodel.name = params.robot_name
        self.x = MX.sym("x", nq * 2)
        self.x_dot = MX.sym("x_dot", nq * 2)
        self.u = MX.sym("u", nu)
        self.p = MX.sym("p", nq)
            
        self.amodel.x = self.x
        self.amodel.u = self.u
        self.amodel.p = self.p

        self.nx = self.amodel.x.size()[0]
        self.nu = self.amodel.u.size()[0]
        self.ny = self.nx + self.nu
        self.nq = nq
        self.nv = nq

        # Rotation matrix 
        euler_angles = self.x[3:6] 
        roll, pitch, yaw = euler_angles[0], euler_angles[1], euler_angles[2]

        R_x = MX([[1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)]])

        R_y = MX([[np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)]])

        R_z = MX([[np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]])

        R_tot = R_z @ R_y @ R_x

        self.Rrpy = Function('R', [self.x], [R_tot])

        # F and M matrices
        sin_a = np.sin(self.alpha)
        cos_a = np.cos(self.alpha)

        self.F = self.cf * np.array([
        [0, np.sqrt(3)/2 * sin_a, -np.sqrt(3)/2 * sin_a, 0, np.sqrt(3)/2 * sin_a, -np.sqrt(3)/2 * sin_a],
        [sin_a, -1/2 * sin_a, -1/2 * sin_a, sin_a, -1/2 * sin_a, -1/2 * sin_a],
        [cos_a, cos_a, cos_a, cos_a, cos_a, cos_a]
        ])

        self.M = self.ct * np.array([
            [0, np.sqrt(3)/2 * self.r * cos_a - np.sqrt(3)/2 * sin_a, np.sqrt(3)/2 * self.r * cos_a - np.sqrt(3)/2 * sin_a, 0, -np.sqrt(3)/2 * self.r * cos_a + np.sqrt(3)/2 * sin_a, -np.sqrt(3)/2 * self.r * cos_a + np.sqrt(3)/2 * sin_a],
            [-self.r * cos_a + sin_a, -1/2 * self.r * cos_a + 1/2 * sin_a, 1/2 * self.r * cos_a - 1/2 * sin_a, self.r * cos_a - sin_a, 1/2 * self.r * cos_a - 1/2 * sin_a, -1/2 * self.r * cos_a + 1/2 * sin_a],
            [self.r * sin_a + cos_a, -self.r * sin_a - cos_a, self.r * sin_a + cos_a, -self.r * sin_a - cos_a, self.r * sin_a + cos_a, -self.r * sin_a - cos_a]
        ])

        # Control force and torque
        self.fc = Function('fc', [self.x, self.u], [self.R(self.x) @ self.F @ self.u])
        self.tc = Function('tc', [self.u], [self.M @ self.u])
        
        self.eps = params.state_tol

class AbstractController:
    def __init__(self, model):
        self.ocp_name = "".join(re.findall('[A-Z][^A-Z]*', self.__class__.__name__)[:-1]).lower()
        self.params = model.params
        self.model = model
        self.obstacles = obstacles  

        self.N = self.params.N
        self.ocp = AcadosOcp()

        # Dimensions
        self.ocp.solver_options.tf = self.params.N * self.params.dt
        self.ocp.dims.N = self.N

        # Model
        self.ocp.model = self.model.amodel

        # Cost
        self.addCost()

        # Constraints
        self.ocp.constraints.lbx_0 = self.model.x_min
        self.ocp.constraints.ubx_0 = self.model.x_max
        self.ocp.constraints.idxbx_0 = np.arange(self.model.nx)

        self.ocp.constraints.lbx = self.model.x_min
        self.ocp.constraints.ubx = self.model.x_max
        self.ocp.constraints.idxbx = np.arange(self.model.nx)

        self.ocp.constraints.lbx_e = self.model.x_min
        self.ocp.constraints.ubx_e = self.model.x_max
        self.ocp.constraints.idxbx_e = np.arange(self.model.nx)

        # Nonlinear constraint 
        self.nl_con_0, self.nl_lb_0, self.nl_ub_0 = [], [], []
        self.nl_con, self.nl_lb, self.nl_ub = [], [], []
        self.nl_con_e, self.nl_lb_e, self.nl_ub_e = [], [], []
        
        # --> dynamics (only on running nodes)
        self.nl_con_0.append(self.model.tau)
        self.nl_lb_0.append(self.model.tau_min)
        self.nl_ub_0.append(self.model.tau_max)
        
        self.nl_con.append(self.model.tau)
        self.nl_lb.append(self.model.tau_min)
        self.nl_ub.append(self.model.tau_max)

        # Additional constraints
        self.addConstraint()
        
        self.model.amodel.con_h_expr_0 = vertcat(*self.nl_con_0)   
        self.model.amodel.con_h_expr = vertcat(*self.nl_con)

        self.ocp.constraints.lh_0 = np.hstack(self.nl_lb_0)
        self.ocp.constraints.uh_0 = np.hstack(self.nl_ub_0)
        self.ocp.constraints.lh = np.hstack(self.nl_lb)
        self.ocp.constraints.uh = np.hstack(self.nl_ub)

        if len(self.nl_con_e) > 0:
            self.model.amodel.con_h_expr_e = vertcat(*self.nl_con_e)
            self.ocp.constraints.lh_e = np.array(self.nl_lb_e)
            self.ocp.constraints.uh_e = np.array(self.nl_ub_e)

        # Solver options
        self.ocp.solver_options.integrator_type = "DISCRETE"
        self.ocp.solver_options.hessian_approx = "EXACT"
        self.ocp.solver_options.exact_hess_constr = 0
        self.ocp.solver_options.exact_hess_dyn = 0
        self.ocp.solver_options.nlp_solver_type = self.params.solver_type
        self.ocp.solver_options.hpipm_mode = self.params.solver_mode
        self.ocp.solver_options.nlp_solver_max_iter = self.params.nlp_max_iter
        self.ocp.solver_options.qp_solver_iter_max = self.params.qp_max_iter
        self.ocp.solver_options.globalization = self.params.globalization
        self.ocp.solver_options.alpha_reduction = self.params.alpha_reduction
        self.ocp.solver_options.alpha_min = self.params.alpha_min
        self.ocp.solver_options.levenberg_marquardt = self.params.levenberg_marquardt

        # Generate OCP solver
        gen_name = self.params.GEN_DIR + 'ocp_' + self.ocp_name + '_' + self.model.amodel.name
        self.ocp.code_export_directory = gen_name
        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file=gen_name + '.json', build=self.params.build)

        # Storage
        self.x_guess = np.zeros((self.N, self.model.nx))
        self.u_guess = np.zeros((self.N, self.model.nu))
        self.tol = self.params.cost_tol

    def addCost(self):
        pass

    def addConstraint(self):
        pass

    def setGuess(self, x_guess, u_guess):
        self.x_guess = x_guess
        self.u_guess = u_guess

    def getGuess(self):
        return np.copy(self.x_guess), np.copy(self.u_guess)

    def resetHorizon(self, N):
        self.N = N
        self.ocp_solver.set_new_time_steps(np.full(N, self.params.dt))
        self.ocp_solver.update_qp_solver_cond_N(N)

    def checkCollision(self, x):
        if self.obstacles is not None and self.params.obs_flag:
            t_glob = self.model.jointToEE(x) 
            for obs in self.obstacles:
                if obs['name'] == 'floor':
                    if t_glob[2] < obs['bounds'][0]:
                        return False
                elif obs['name'] == 'ball':
                    dist_b = np.sum((t_glob.flatten() - obs['position']) ** 2) 
                    if dist_b < obs['bounds'][0]:
                        return False
        return True