import re
import numpy as np
from casadi import (
    MX, DM, horzcat, vertcat, dot, Function,
    sin, cos, tan, cross, fabs, sqrt, diag
)
from urdf_parser_py.urdf import URDF
import adam
from adam.casadi import KinDynComputations
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

class Model:
    """Multirotor rigid-body model with CasADi symbolic expressions for acados.

    Builds the full symbolic state-space representation of a Star-shaped
    Tilted Hexarotor (STH), including rotation matrices, thrust/torque
    allocation matrices, explicit dynamics, state/input bounds, and the
    nonlinear obstacle-avoidance constraints required by the OCP.

    Parameters
    ----------
    params : Parameters
        Configuration object containing all physical and solver parameters.
    """
    def __init__(self, params: object) -> None:
        self.params = params
        
        # --- Physical parameters ---
        self.mass = params.mass
        self.J = params.J
        self.l = params.l 
        self.cf = params.cf
        self.ct = params.ct
        self.r = self.cf / self.ct * self.l # Effective moment arm ratio
        self.g = 9.81   # Gravitational acceleration [m/s²]
        self.u_bar = params.u_bar 
        self.alpha_tilt = params.alpha_tilt
        self.eps = params.state_tol

        # --- Environment and drone size bounds ---
        self.min_width = params.min_width
        self.min_length = params.min_length
        self.min_height = params.min_height
        self.max_width = params.max_width
        self.max_length = params.max_length
        self.max_height = params.max_height
        self.v_min = params.v_min
        self.v_max = params.v_max
        

        # --- State and input dimensions ---
        nq = 6  # Pose: 3 position + 3 orientation (Euler angles)
        nu = 6  # Input: 6 squared rotor spinning rates
        npos = 3    # Position sub-space dimension
        nori = 3    # Orientation sub-space dimension
        nbox = 6    # Obstacle box constraint dimension

        # --- CasADi symbolic variables ---
        self.x = MX.sym("x", nq * 2)    # Full state [q; q_dot]
        self.x_dot = MX.sym("x_dot", nq * 2)
        self.u = MX.sym("u", nu)    # Control input
        self.p = MX.sym("p", nq)    # OCP parameter (direction vector)
            
        # --- Rotation matrix (ZYX Euler convention) ---
        euler_angles = self.x[npos:npos+nori] 
        roll, pitch, yaw = euler_angles[0], euler_angles[1], euler_angles[2]

        R_x = vertcat(
            horzcat(1,          0,          0   ),
            horzcat(0,  cos(roll),  -sin(roll)  ),
            horzcat(0,  sin(roll),   cos(roll)   )
        )
        R_y = vertcat(
            horzcat( cos(pitch),     0,     sin(pitch)),
            horzcat(          0,     1,              0),
            horzcat(-sin(pitch),     0,     cos(pitch))
        )
        R_z = vertcat(
            horzcat(cos(yaw), -sin(yaw), 0),
            horzcat(sin(yaw),  cos(yaw), 0),
            horzcat(       0,         0, 1)
        )

        # Total rotation matrix: world ← body
        R_tot = R_z @ R_y @ R_x
        self.R = Function('R', [self.x], [R_tot])

        # --- Thrust and torque allocation matrices ---
        sin_a = np.sin(self.alpha_tilt)
        cos_a = np.cos(self.alpha_tilt)
        tan_a = np.tan(self.alpha_tilt)

        # F maps squared rotor speeds to body-frame force components [3 × nu]
        self.F = self.cf * np.array([
            [0, np.sqrt(3)/2 * sin_a, -np.sqrt(3)/2 * sin_a, 
             0, np.sqrt(3)/2 * sin_a, -np.sqrt(3)/2 * sin_a],
            [sin_a, -1/2 * sin_a, -1/2 * sin_a, 
             sin_a, -1/2 * sin_a, -1/2 * sin_a],
            [cos_a, cos_a, cos_a, 
             cos_a, cos_a, cos_a]
        ])

        # M maps squared rotor speeds to body-frame torque components [3 × nu]
        self.M = self.ct * np.array([
            [0, 
             np.sqrt(3)/2 * self.r * cos_a - np.sqrt(3)/2 * sin_a, 
             np.sqrt(3)/2 * self.r * cos_a - np.sqrt(3)/2 * sin_a, 
             0, 
             -np.sqrt(3)/2 * self.r * cos_a + np.sqrt(3)/2 * sin_a, 
             -np.sqrt(3)/2 * self.r * cos_a + np.sqrt(3)/2 * sin_a],
            [-self.r * cos_a + sin_a, 
             -1/2 * self.r * cos_a + 1/2 * sin_a, 
              1/2 * self.r * cos_a - 1/2 * sin_a, 
              self.r * cos_a - sin_a, 
              1/2 * self.r * cos_a - 1/2 * sin_a,
             -1/2 * self.r * cos_a + 1/2 * sin_a],
            [self.r * sin_a + cos_a, 
             -self.r * sin_a - cos_a, 
              self.r * sin_a + cos_a, 
             -self.r * sin_a - cos_a, 
              self.r * sin_a + cos_a, 
             -self.r * sin_a - cos_a]
        ])

        # CasADi functions for net control force and torque
        self.fc = Function(
            'fc', 
            [self.x, self.u], 
            [self.R(self.x) @ self.F @ self.u]
        )
        self.tc = Function(
            'tc', 
            [self.u], 
            [self.M @ self.u]
        )

        # --- Euler-rate to angular-velocity transformation (inverse) ---
        Tinv_expr = vertcat(
            horzcat(1, sin(roll)*tan(pitch),  cos(roll)*tan(pitch)),
            horzcat(0,            cos(roll),            -sin(roll)),
            horzcat(0, sin(roll)/cos(pitch),  cos(roll)/cos(pitch))
        )
        self.Tinv = Function('Tinv', [self.x], [Tinv_expr])

        # --- Explicit continuous-time dynamics: x_dot = f(x, u) ---
        self.f_expl = vertcat(
            self.x[nq:nq+npos],
            self.Tinv(self.x)@self.x[nq+npos:],
            -self.g*np.array([[0], [0], [1]]) + self.fc(self.x, self.u) /
            self.mass, 
            np.linalg.inv(self.J) @ (
                -cross(self.x[nq+npos:], self.J @ self.x[nq+npos:])) + 
                np.linalg.inv(self.J) @ self.tc(self.u)
        )

        # --- Input bounds ---
        self.u_max = np.full(nu, self.u_bar)
        self.u_min = np.zeros((nu,))

        # --- Feasible orientation bounds (derived from actuator saturation) 
        # ---
        # Three regimes depending on the ratio between u_bar and hovering thrust
        if self.u_bar >= (self.mass*self.g)/(2*self.cf*cos_a):
            ri = abs(-self.mass*self.g/2 * tan_a)
            ro = self.mass*self.g * tan_a
        elif (self.mass*self.g) / (4*self.cf*cos_a) <= self.u_bar \
              < (self.mass*self.g) / (2*self.cf*cos_a):
            ri = min(
                abs(-self.mass*self.g/2 * tan_a), 
                abs(3*self.cf*self.u_bar*sin_a -self.mass*self.g/2 * tan_a)
            )   
            ro = np.linalg.norm(np.array([
                np.sqrt(3)*self.cf*self.u_bar*sin_a, 
                self.mass*self.g * tan_a - 3*self.cf*self.u_bar*sin_a
            ]))
        else:
            ri = 3*self.cf*self.u_bar*sin_a - self.mass*self.g/2 * tan_a
            ro = self.mass * self.g * tan_a - 6* self.cf * self.u_bar * sin_a

        # Maximum tilt angles admissible for hovering and full actuation
        self.phi_hovering = np.arctan2(ri, self.mass * self.g)
        self.phi_hovering_max = np.arctan2(ro, self.mass * self.g) 
        self.phi_max = np.arccos(
            (self.mass*self.g) / 
            (self.cf * 6 * np.cos(self.alpha_tilt)*self.u_bar)
        )

        # --- Ellipsoidal obstacle-avoidance constraint ---
        # Q(x) = R(x) · diag(half-axes²) · R(x)ᵀ encodes drone occupancy as
        # a rotation-aware ellipsoid; constraint: nᵢᵀ p + √(nᵢᵀ Q nᵢ) ≤ bound_i
        D = diag(vertcat(
            self.min_width**2, 
            self.min_length**2, 
            self.min_height**2
        ))
        self.Q = Function(
            'Q', 
            [self.x], 
            [self.R(self.x) @ D @ self.R(self.x).T]
        )

        # Outward normals of the six faces of the axis-aligned bounding box
        box_normals = [
            DM([ 1.0,  0.0,  0.0]), # left
            DM([ 0.0,  1.0,  0.0]), # back
            DM([ 0.0,  0.0,  1.0]), # bottom
            DM([-1.0,  0.0,  0.0]), # right
            DM([ 0.0, -1.0,  0.0]), # front
            DM([ 0.0,  0.0, -1.0]), # top
        ]

        # Build one scalar constraint per face
        self.con_h_expr_list = []
        for n in box_normals:
            expr = n.T @ self.x[:npos] + sqrt(n.T @ self.Q(self.x) @ n)
            self.con_h_expr_list.append(expr)

        self.con_h_expr = vertcat(*self.con_h_expr_list)

        # Signed environment extents used as constraint bounds [m]
        self.env_dimensions = np.array([
            -self.max_width, -self.max_length, -self.max_height,
             self.max_width, self.max_length, self.max_height
        ]) 
        
        # Signed drone half-extents (same sign convention) [m]
        self.drone_occupancy = np.array([
            -self.min_width, -self.min_length, -self.min_height,
             self.min_width, self.min_length, self.min_height
        ]) 

        # --- Acados model registration ---
        self.amodel = AcadosModel()
        self.amodel.name = params.robot_name
        self.amodel.x = self.x
        self.amodel.u = self.u
        self.amodel.f_expl_expr = self.f_expl
        self.amodel.p = self.p
        
        # Derived dimension attributes exposed to controllers
        self.nx = self.amodel.x.size()[0]   # Full state dimension
        self.nu = self.amodel.u.size()[0]   # Input dimension
        self.ny = self.nx + self.nu         # Output dimension (state + input)
        self.nq = nq                        # Generalised coordinate dimension
        self.nv = nq                        # Velocity dimension (= nq for 
                                            # this model)
        self.npos = npos                    # Position sub-space dimension
        self.nori = nori                    # Orientation sub-space dimension
        self.nbox = self.con_h_expr.size()[0]  # Number of box-face constraints

class AbstractController:
    """Base class for acados-based OCP controllers.

    Builds and compiles a shared AcadosOcp object from the supplied model and
    parameter set. Subclasses implement specific solve strategies on top of
    the common solver infrastructure created here.

    Parameters
    ----------
    model : Model
        Symbolic robot model providing dynamics, constraints, and dimensions.
    """

    def __init__(self, model: object) -> None:
        # Derive a human-readable OCP name from the concrete subclass name
        self.ocp_name = "".join(
            re.findall('[A-Z][^A-Z]*', self.__class__.__name__)[:-1]).lower()
        self.params = model.params
        self.model = model

        self.N = self.params.N
        self.ocp = AcadosOcp()

        # --- Horizon and time discretisation ---
        self.ocp.solver_options.tf = self.params.N * self.params.dt
        self.ocp.dims.N = self.N

        # --- Symbolic model ---
        self.ocp.model = self.model.amodel

        # --- Stage cost: maximise initial velocity projected onto d ---
        self.ocp.cost.cost_type_0 = 'EXTERNAL'
        self.ocp.model.cost_expr_ext_cost_0 = -dot(
            self.model.p[:self.model.nq], 
            self.model.x[self.model.nq:]
        )
        self.ocp.parameter_values = np.zeros(self.model.nv)

        # --- Initial shooting node: position fixed, velocity free ---
        self.ocp.constraints.lbx_0 = np.full(self.model.nq, 0.0) 
        self.ocp.constraints.ubx_0 = np.full(self.model.nq, 0.0) 
        self.ocp.constraints.idxbx_0 = np.arange(self.model.nq)       

        # --- Path constraints ---
        # Nonlinear obstacle constraint (one per box face)
        self.ocp.model.con_h_expr = self.model.con_h_expr 
        self.ocp.constraints.uh = np.full(self.model.nbox, 0.0)
        self.ocp.constraints.lh = np.full(self.model.nbox, -1e2)

        # State box: orientation in [-π, π], velocity loosely bounded
        self.ocp.constraints.lbx = np.hstack([
            np.full(self.model.nori, -np.pi), 
            np.full(self.model.nv, -1e2)
        ])
        self.ocp.constraints.ubx = np.hstack([
            np.full(self.model.nori, np.pi), 
            np.full(self.model.nv, 1e2)]) 
        self.ocp.constraints.idxbx = np.arange(
            self.model.npos, self.model.nx
        )       

        # --- Terminal constraints: zero orientation and velocity ---
        self.ocp.constraints.lbx_e = np.full(
            self.model.nori + self.model.nv, 0.0
        ) 
        self.ocp.constraints.ubx_e = np.full(
            self.model.nori + self.model.nv, 0.0
        )  
        self.ocp.constraints.idxbx_e = np.arange(
            self.model.npos, self.model.nx
        )      

        # Terminal obstacle constraint (same expression as path)
        self.ocp.model.con_h_expr_e = self.model.con_h_expr 
        self.ocp.constraints.uh_e = np.full(self.model.nbox, 0.0)
        self.ocp.constraints.lh_e = np.full(self.model.nbox, -1e2)

        # Linear equality constraint C·x + D·u ∈ [lg, ug] (used for velocity projection)
        self.ocp.constraints.C = np.zeros((self.model.nv, self.model.nx))
        self.ocp.constraints.D = np.zeros((self.model.nv, self.model.nu))
        self.ocp.constraints.lg = np.zeros((self.model.nv,))
        self.ocp.constraints.ug = np.zeros((self.model.nv,))

        # --- Input bounds: rotor speeds non-negative and below saturation ---
        self.ocp.constraints.lbu = self.model.u_min
        self.ocp.constraints.ubu = self.model.u_max
        self.ocp.constraints.idxbu = np.arange(self.model.nu)

        # --- Solver options ---
        self.ocp.solver_options.integrator_type = "ERK"
        self.ocp.solver_options.hessian_approx = "EXACT"
        self.ocp.solver_options.exact_hess_constr = 0   # Constraint Hessian 
                                                        # approximated
        self.ocp.solver_options.exact_hess_dyn = 0  # Dynamics Hessian 
                                                    # approximated
        self.ocp.solver_options.nlp_solver_type = self.params.solver_type
        self.ocp.solver_options.hpipm_mode = self.params.solver_mode
        self.ocp.solver_options.nlp_solver_max_iter = self.params.nlp_max_iter
        self.ocp.solver_options.qp_solver_iter_max = self.params.qp_max_iter
        self.ocp.solver_options.globalization = self.params.globalization
        self.ocp.solver_options.globalization_alpha_reduction = self.params.alpha_reduction
        self.ocp.solver_options.globalization_alpha_min = self.params.alpha_min
        self.ocp.solver_options.levenberg_marquardt = self.params.levenberg_marquardt
        self.ocp.solver_options.tol = self.params.state_tol
        self.ocp.solver_options.print_level = 0  # Suppress solver output

        # --- Code generation and solver compilation ---
        gen_name = self.params.GEN_DIR + 'ocp_' + self.ocp_name + '_' + self.model.amodel.name
        self.ocp.code_export_directory = gen_name
        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file=gen_name + '.json', build=self.params.build)

        # --- Warm-start storage ---
        self.x_guess = np.zeros((self.N, self.model.nx))
        self.u_guess = np.zeros((self.N, self.model.nu))
        self.tol = self.params.cost_tol

    def setGuess(
        self,
        x_guess: np.ndarray,
        u_guess: np.ndarray,
    ) -> None:
        """Store a new warm-start trajectory for the next solver call.

        Parameters
        ----------
        x_guess : np.ndarray
            State trajectory used as initial guess, shape (N, nx).
        u_guess : np.ndarray
            Input trajectory used as initial guess, shape (N, nu).
        """
        self.x_guess = x_guess
        self.u_guess = u_guess

    def getGuess(self) -> tuple[np.ndarray, np.ndarray]:
        """Return a copy of the current warm-start trajectory.

        Returns
        -------
        x_guess : np.ndarray
            Copy of the stored state trajectory, shape (N, nx).
        u_guess : np.ndarray
            Copy of the stored input trajectory, shape (N, nu).
        """
        return np.copy(self.x_guess), np.copy(self.u_guess)

    def resetHorizon(self, N: int) -> None:
        """Update the solver horizon length and re-condition the QP.

        Parameters
        ----------
        N : int
            New prediction horizon length (number of shooting intervals).
        """
        self.N = N
        self.ocp_solver.set_new_time_steps(np.full(N, self.params.dt))
        self.ocp_solver.update_qp_solver_cond_N(N)