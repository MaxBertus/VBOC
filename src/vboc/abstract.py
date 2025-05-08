import re
import numpy as np
from casadi import MX, DM, horzcat, vertcat, dot, Function, sin, cos, tan, cross, fabs, sqrt, diag
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
        self.r = self.cf / self.ct * self.l
        self.g = 9.81
        self.u_bar = params.u_bar 
        self.alpha_tilt = params.alpha_tilt
        self.min_width = params.min_width
        self.min_length = params.min_length
        self.min_height = params.min_height
        self.max_width = params.max_width
        self.max_length = params.max_length
        self.max_height = params.max_height
        self.eps = params.state_tol

        nq = 6 # dimension of pose: 3 for position, 3 for orientation (Euler Angles) 
        nu = 6 # dimension of input: 6 squared spinning rates
        npos = 3 # dimension of positon
        nori = 3 # dimension of orientation

        self.x = MX.sym("x", nq * 2)
        self.x_dot = MX.sym("x_dot", nq * 2)
        self.u = MX.sym("u", nu)
        self.p = MX.sym("p", nq)
            
        # Rotation matrix 
        euler_angles = self.x[npos:npos+nori] 
        roll, pitch, yaw = euler_angles[0], euler_angles[1], euler_angles[2]

        R_x = vertcat(
            horzcat(1, 0, 0),
            horzcat(0, cos(roll), -sin(roll)),
            horzcat(0, sin(roll), cos(roll)))

        R_y = vertcat(
                horzcat(cos(pitch), 0, sin(pitch)),
                horzcat(0, 1, 0),
                horzcat(-sin(pitch), 0, cos(pitch)))
        
        R_z = vertcat(
            horzcat(cos(yaw), -sin(yaw), 0),
            horzcat(sin(yaw), cos(yaw), 0),
            horzcat(0, 0, 1))

        R_tot = R_z @ R_y @ R_x

        self.R = Function('R', [self.x], [R_tot])

        # F and M matrices
        sin_a = np.sin(self.alpha_tilt)
        cos_a = np.cos(self.alpha_tilt)
        tan_a = np.tan(self.alpha_tilt)

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

        Tinv_expr = vertcat(
            horzcat(1, sin(roll)*tan(pitch), cos(roll)*tan(pitch)),
            horzcat(0, cos(roll), -sin(roll)),
            horzcat(0, sin(roll)/cos(pitch), cos(roll)/cos(pitch)))

        self.Tinv = Function('Tinv', [self.x], [Tinv_expr])

        # explicit dynamics
        self.f_expl = vertcat(
            self.x[nq:nq+npos],
            self.Tinv(self.x)@self.x[nq+npos:],
            -self.g*np.array([[0], [0], [1]]) + self.fc(self.x, self.u)/self.mass, 
            np.linalg.inv(self.J) @ (cross(self.x[nq+npos:], self.J @ self.x[nq+npos:])) + np.linalg.inv(self.J) @ self.tc(self.u)
        )

        # explicit dynamics function
        # self.f_expl_func = Function('f_expl', [self.x, self.u], [self.f_expl])

        # BOUNDS
        # Input 
        self.u_max = np.array([self.u_bar, self.u_bar, self.u_bar, self.u_bar, self.u_bar, self.u_bar])
        self.u_min = np.zeros((nu,))

        # Orientation
        if self.u_bar >= (self.mass*self.g)/(2*self.cf*cos_a):
            ri = abs(-self.mass*self.g/2 * tan_a)
            ro = self.mass*self.g * tan_a
        elif (self.mass*self.g)/(4*self.cf*cos_a) <= self.u_bar < (self.mass*self.g)/(2*self.cf*cos_a):
            ri = min(abs(-self.mass*self.g/2 * tan_a), abs(3*self.cf*self.u_bar*sin_a -self.mass*self.g/2 * tan_a))   
            ro = np.linalg.norm(np.array([np.sqrt(3)*self.cf*self.u_bar*sin_a, self.mass*self.g * tan_a - 3*self.cf*self.u_bar*sin_a]))
        else:
            ri = 3*self.cf*self.u_bar*sin_a - self.mass*self.g/2 * tan_a
            ro = self.mass * self.g * tan_a - 6* self.cf * self.u_bar * sin_a

        self.phi_hovering = np.arctan2(ri, self.mass * self.g) # max inclination allowed for hovering
        self.phi_hovering_max = np.arctan2(ro, self.mass * self.g) # max inclination allowed for hovering 
        self.phi_max = np.arccos((self.mass*self.g)/(self.cf * 6 * np.cos(self.alpha_tilt)*self.u_bar))

        # Position
        # Define symbolic parameters for the box bounds
        # self.box_min = MX.sym("box_min", 3)  # [box_min_x, box_min_y, box_min_z]
        # self.box_max = MX.sym("box_max", 3)  # [box_max_x, box_max_y, box_max_z]

        # self.box_occupancy = np.array([-self.min_width, -self.min_length, -self.min_height,
        #                                 self.min_width, self.min_length, self.min_height]) 

        D = diag(vertcat(self.min_width**2, self.min_length**2, self.min_height**2))
        self.Q = Function('Q', [self.x], [self.R(self.x) @ D @ self.R(self.x).T])

        box_normals = [
            DM([1.0, 0.0, 0.0]),   # left
            DM([0.0, 1.0, 0.0]),   # back
            DM([0.0, 0.0, 1.0]),   # bottom
            DM([-1.0, 0.0, 0.0]),  # right
            DM([0.0, -1.0, 0.0]),  # front
            DM([0.0, 0.0, -1.0]),  # top
        ]

        self.con_h_expr_list = []

        for n in box_normals:
            expr = n.T @ self.x[:npos] + sqrt(n.T @ self.Q(self.x) @ n)
            print(n.T @  self.x[:npos])
            self.con_h_expr_list.append(expr)

        self.con_h_expr = vertcat(*self.con_h_expr_list)

        # self.env_dimensions = np.array([-self.max_width, -self.max_length, -self.max_height,
        #                                 self.max_width, self.max_length, self.max_height]) 

        # Acados model
        self.amodel = AcadosModel()
        self.amodel.name = params.robot_name
        self.amodel.x = self.x
        self.amodel.u = self.u
        self.amodel.f_expl_expr = self.f_expl
        self.amodel.p = self.p
        
        self.nx = self.amodel.x.size()[0]
        self.nu = self.amodel.u.size()[0]
        self.ny = self.nx + self.nu
        self.nq = nq
        self.nv = nq
        self.npos = npos
        self.nori = nori
        self.nbox = self.con_h_expr.size()[0]

class AbstractController:
    def __init__(self, model):
        self.ocp_name = "".join(re.findall('[A-Z][^A-Z]*', self.__class__.__name__)[:-1]).lower()
        self.params = model.params
        self.model = model

        self.N = self.params.N
        self.ocp = AcadosOcp()

        # DIMENSIONS
        self.ocp.solver_options.tf = self.params.N * self.params.dt
        self.ocp.dims.N = self.N

        # MODEL
        self.ocp.model = self.model.amodel

        # COST
        # Maximize initial velocity
        self.ocp.cost.cost_type_0 = 'EXTERNAL'
        self.ocp.model.cost_expr_ext_cost_0 =  dot(self.model.p[:self.model.nq], self.model.x[self.model.nq:])
        self.ocp.parameter_values = np.zeros(self.model.nv)

        # CONSTRAINTS
        # Initial shooting node constraints
        self.ocp.constraints.lbx_0 = np.full(self.model.nq, np.zeros(self.model.nq))  
        self.ocp.constraints.ubx_0 = np.full(self.model.nq, np.zeros(self.model.nq))  
        self.ocp.constraints.idxbx_0 = np.arange(self.model.nq)       

        # Path constraints
        self.ocp.model.con_h_expr = self.model.con_h_expr 
        self.ocp.constraints.uh = np.full(self.model.nbox, 0.0)
        self.ocp.constraints.lh = np.full(self.model.nbox, -1e2)

        # Terminal constraints
        self.ocp.constraints.lbx_e = np.full(self.model.nv, np.zeros(self.model.nv))  
        self.ocp.constraints.ubx_e = np.full(self.model.nv, np.zeros(self.model.nv))  
        self.ocp.constraints.idxbx_e = np.arange(self.model.nq, self.model.nx)      

        self.ocp.model.con_h_expr_e = self.model.con_h_expr 
        self.ocp.constraints.uh_e = np.full(self.model.nbox, 0.0)
        self.ocp.constraints.lh_e = np.full(self.model.nbox, -1e2)

        self.ocp.constraints.C = np.zeros((self.model.nv, self.model.nx))
        self.ocp.constraints.D = np.zeros((self.model.nv, self.model.nu))
        self.ocp.constraints.lg = np.zeros((self.model.nv,))
        self.ocp.constraints.ug = np.zeros((self.model.nv,))

        # Input constraints
        self.ocp.constraints.lbu = self.model.u_min
        self.ocp.constraints.ubu = self.model.u_max
        self.ocp.constraints.idxbu = np.arange(self.model.nu)

        # SOLVER OPTIONS
        self.ocp.solver_options.integrator_type = "ERK"
        self.ocp.solver_options.hessian_approx = "EXACT"
        self.ocp.solver_options.exact_hess_constr = 0
        self.ocp.solver_options.exact_hess_dyn = 0
        self.ocp.solver_options.nlp_solver_type = self.params.solver_type
        self.ocp.solver_options.hpipm_mode = self.params.solver_mode
        self.ocp.solver_options.nlp_solver_max_iter = self.params.nlp_max_iter
        self.ocp.solver_options.qp_solver_iter_max = self.params.qp_max_iter
        self.ocp.solver_options.globalization = self.params.globalization
        self.ocp.solver_options.globalization_alpha_reduction = self.params.alpha_reduction
        self.ocp.solver_options.globalization_alpha_min = self.params.alpha_min
        self.ocp.solver_options.levenberg_marquardt = self.params.levenberg_marquardt

        # Debug
        self.ocp.solver_options.print_level = 0

        # Generate OCP solver
        gen_name = self.params.GEN_DIR + 'ocp_' + self.ocp_name + '_' + self.model.amodel.name
        self.ocp.code_export_directory = gen_name
        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file=gen_name + '.json', build=self.params.build)

        # Storage
        self.x_guess = np.zeros((self.N, self.model.nx))
        self.u_guess = np.zeros((self.N, self.model.nu))
        self.tol = self.params.cost_tol

    def setGuess(self, x_guess, u_guess):
        self.x_guess = x_guess
        self.u_guess = u_guess

    def getGuess(self):
        return np.copy(self.x_guess), np.copy(self.u_guess)

    def resetHorizon(self, N):
        self.N = N
        self.ocp_solver.set_new_time_steps(np.full(N, self.params.dt))
        self.ocp_solver.update_qp_solver_cond_N(N)