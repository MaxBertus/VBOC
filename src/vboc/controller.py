import numpy as np
from casadi import dot, horzcat
from .abstract import AbstractController


class ViabilityController(AbstractController):
    """Controller that solves a Viability-Based Optimal Control (VBOC) problem.

    Inherits from AbstractController and extends it with methods to solve
    a single OCP instance and an iterative horizon-expansion procedure
    for computing backward reachable sets.

    Parameters
    ----------
    model : object
        Robot model exposing state/input dimensions and constraint metadata.
    """

    def __init__(self, model) -> None:
        super().__init__(model)
        # Projection matrix used to enforce the velocity constraint orthogonal 
        # to d
        self.C = np.zeros((self.model.nv, self.model.nx))

    def solve(
        self,
        q_init: np.ndarray,
        d: np.ndarray,
        box_min_values: np.ndarray,
        box_max_values: np.ndarray,
    ) -> int:
        """Set up and solve a single OCP instance for a given initial 
        configuration.

        Initialises all stage variables with the current warm-start guess,
        applies box constraints on position and velocity at every interior
        stage, enforces zero-velocity terminal constraints at stages N-1 and N,
        and fixes the initial position to ``q_init`` while leaving the initial
        velocity free (subject to the orthogonality constraint w.r.t. ``d``).

        Parameters
        ----------
        q_init : np.ndarray
            Initial generalised position, shape (nq,).
        d : np.ndarray
            Direction vector used to project the velocity space.
        box_min_values : np.ndarray
            Lower bounds for the obstacle-box constraint.
        box_max_values : np.ndarray
            Upper bounds for the obstacle-box constraint.

        Returns
        -------
        status : int
            Solver status code (0 = success, 2 = acceptable solution).
        """
        self.ocp_solver.reset()
        
        for i in range(self.N):
            # Warm-start primal variables and set the direction parameter
            self.ocp_solver.set(i, 'x', self.x_guess[i])
            self.ocp_solver.set(i, 'u', self.u_guess[i])
            self.ocp_solver.set(i, 'p', d)

            if i != 0:
                # Box constraint on the obstacle (upper bound encodes both min 
                # and max)
                self.ocp_solver.constraints_set(
                     i, "uh", 
                     np.hstack([box_min_values, box_max_values])
                )
                # Orientation bounded in [-π, π]; velocity loosely bounded for 
                # feasibility
                self.ocp_solver.constraints_set(
                     i, 
                     "lbx", 
                     np.hstack([
                          np.full(self.model.nori, -np.pi), 
                          np.full(self.model.nv, -1e2)
                     ])
                )
                self.ocp_solver.constraints_set(
                     i, 
                     "ubx", 
                     np.hstack([
                          np.full(self.model.nori, np.pi), 
                          np.full(self.model.nv, 1e2)
                    ])
                )

        # Warm-start terminal stage
        self.ocp_solver.set(self.N, 'x', self.x_guess[-1])
        self.ocp_solver.set(self.N, 'p', d)

        # Shared bounds reused for stages N-1 and N
        zero_vel = np.zeros(self.model.nv)
        ori_lb = np.full(self.model.nori, -np.pi)
        ori_ub = np.full(self.model.nori, np.pi)

        # Zero-velocity terminal constraint at stage N-1 (orientation free in [-π, π])
        self.ocp_solver.constraints_set(
            self.N - 1, "lbx", np.hstack([ori_lb, zero_vel])
        )
        self.ocp_solver.constraints_set(
            self.N - 1, "ubx", np.hstack([ori_ub, zero_vel])
        )

        # Zero-velocity terminal constraint at final stage N
        self.ocp_solver.constraints_set(
            self.N, "lbx", np.hstack([ori_lb, zero_vel])
        )
        self.ocp_solver.constraints_set(
            self.N, "ubx", np.hstack([ori_ub, zero_vel])
        )
        self.ocp_solver.constraints_set(
            self.N, "uh",
            np.hstack([box_min_values, box_max_values])
        )

        # Build the projection matrix C = [0 | I - d·dᵀ] to constrain
        # the initial velocity to lie in the hyperplane orthogonal to d
        d_arr = np.array([d.tolist()])
        self.C[:, self.model.nq:] = (
             np.eye(self.model.nv) - np.matmul(d_arr.T, d_arr)
        )
        self.ocp_solver.constraints_set(0, "C", self.C, api='new')

        # Fix initial position to q_init; initial velocity remains free
        self.ocp_solver.constraints_set(0, "lbx", q_init)
        self.ocp_solver.constraints_set(0, "ubx", q_init)

        return self.ocp_solver.solve()
    
        
    def solve_vboc(
        self,
        q_init: np.ndarray,
        d: np.ndarray,
        box_min_values: np.ndarray,
        box_max_values: np.ndarray,
        N_start: int,
        n: int = 1,
        repeat: int = 10,
    ) -> tuple[np.ndarray | None, np.ndarray | None, int | None, int]:
        """Iteratively expand the horizon to compute the viability kernel boundary.

        Starting from horizon ``N_start``, the method repeatedly solves the OCP,
        rolls out the solution as the new warm-start, and increments the horizon
        by ``n`` steps. The loop terminates early if the projected velocity
        ``gamma`` stops improving by more than ``self.tol``.

        Parameters
        ----------
        q_init : np.ndarray
            Initial generalised position, shape (nq,).
        d : np.ndarray
            Direction vector for the velocity projection.
        box_min_values : np.ndarray
            Lower bounds for the obstacle-box constraint.
        box_max_values : np.ndarray
            Upper bounds for the obstacle-box constraint.
        N_start : int
            Initial prediction horizon length.
        n : int, optional
            Number of steps by which the horizon is incremented each iteration.
            Default is 1.
        repeat : int, optional
            Maximum number of horizon-expansion iterations. Default is 10.

        Returns
        -------
        x_sol : np.ndarray or None
            State trajectory of shape (N+n, nx), or None on failure.
        u_sol : np.ndarray or None
            Input trajectory of shape (N+n, nu), or None on failure.
        N : int or None
            Final horizon length, or None on failure.
        status : int
            Solver status code of the last solve call (0 = success).
        """
        N = N_start
        gamma = 0   # Best projected velocity magnitude found so far
        x_sol, u_sol = None, None

        for r in range(repeat):
            status = self.solve(q_init, d, box_min_values, box_max_values)

            if status == 0 or status == 2:
                # Evaluate the projected velocity at the initial stage
                x0 = self.ocp_solver.get(0, "x")
                gamma_new = d @ x0[self.model.nq:]

                print(
                    f"Iteration {r}: gamma_new = {gamma_new:.4f}, "
                    f"gamma = {gamma:.4f}, "
                    f"diff = {gamma_new - gamma:.4f}, "
                    f"status = {status}"
                )

                # Early stopping: no meaningful improvement and solver 
                # converged
                if gamma_new < gamma - self.tol and status == 0:
                    break
                
                gamma = gamma_new

                # Roll out the full solution trajectory
                x_sol = np.empty((N + n, self.model.nx))
                u_sol = np.empty((N + n, self.model.nu)) 
                for i in range(N):
                    x_sol[i] = self.ocp_solver.get(i, 'x')
                    u_sol[i] = self.ocp_solver.get(i, 'u')

                # Extend trajectory by repeating the terminal state with zero 
                # input
                x_sol[N:] = self.ocp_solver.get(N, 'x')
                u_sol[N:] = np.zeros((n, self.model.nu))

                # Warm-start the next iteration with the current solution
                self.setGuess(x_sol, u_sol)
                
                # Increment the horizon and resize the solver accordingly
                N += n
                self.resetHorizon(N)
                
            else:     
                # Solver failed: abort and propagate the failure status
                return None, None, None, status
            
        if status == 0:
            return x_sol, u_sol, N, status
        else:
            return None, None, None, status