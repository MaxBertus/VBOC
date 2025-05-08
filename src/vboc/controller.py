import numpy as np
from casadi import dot, horzcat
from .abstract import AbstractController


class ViabilityController(AbstractController):
    def __init__(self, model):
        super().__init__(model)
        self.C = np.zeros((self.model.nv, self.model.nx))

    def solve(self, q_init, d, box_min_values, box_max_values):
        self.ocp_solver.reset()
        for i in range(self.N):
            self.ocp_solver.set(i, 'x', self.x_guess[i])
            self.ocp_solver.set(i, 'u', self.u_guess[i])
            self.ocp_solver.set(i, 'p', d)
            if i != 0:
                self.ocp_solver.constraints_set(i, "lbx", box_min_values)
                self.ocp_solver.constraints_set(i, "ubx", box_max_values)
        self.ocp_solver.set(self.N, 'x', self.x_guess[-1]) # NOTE: why?
        self.ocp_solver.constraints_set(self.N, "lbx", np.hstack([box_min_values, np.full(self.model.nori, -1e4), np.zeros((self.model.nv,))]))
        self.ocp_solver.constraints_set(self.N, "ubx", np.hstack([box_max_values, np.full(self.model.nori, 1e4), np.zeros((self.model.nv,))]))
        self.ocp_solver.set(self.N, 'p', d)

        # Set the initial constraint
        d_arr = np.array([d.tolist()])
        self.C[:, self.model.nq:] = np.eye(self.model.nv) - np.matmul(d_arr.T, d_arr)
        self.ocp_solver.constraints_set(0, "C", self.C, api='new')

        # Set initial bounds -> x0_pos = q_init, x0_vel free; (final bounds already set)
        self.ocp_solver.constraints_set(0, "lbx", q_init)
        self.ocp_solver.constraints_set(0, "ubx", q_init)
        # print("lbx", q_init_lb)
        # print("ubx", q_init_ub)

        # Solve the OCP
        return self.ocp_solver.solve()
    
    # def solveVBOC(self, q_init, d, box_min_values, box_max_values, N_start, n=1, repeat=10):
    #     N = N_start
    #     gamma = 0
    #     x_sol, u_sol = None, None

    #     status = self.solve(q_init, d, box_min_values, box_max_values)

    #     if status == 0:
    #         x_sol = np.empty((N + n, self.model.nx))
    #         u_sol = np.empty((N + n, self.model.nu))    # last control is not used
    #         for i in range(N):
    #             x_sol[i] = self.ocp_solver.get(i, 'x')
    #             u_sol[i] = self.ocp_solver.get(i, 'u')
    #         x_sol[N:] = self.ocp_solver.get(N, 'x')
    #         u_sol[N:] = np.zeros((n, self.model.nu))

    #     return x_sol, u_sol, N, status
        
    def solveVBOC(self, q_init, d, box_min_values, box_max_values, N_start, n=1, repeat=10):
        N = N_start
        gamma = 0
        x_sol, u_sol = None, None
        # if n == 0:
        #     # N-BRS --> constant horizon N, no need to repeat the process until convergence 
        #     repeat = 1
        for r in range(repeat):
            
            # Solve the OCP
            status = self.solve(q_init, d, box_min_values, box_max_values)

            if status == 0 or status == 2:
                # Compare the current cost with the previous one:
                x0 = self.ocp_solver.get(0, "x")
                gamma_new = np.linalg.norm(x0[self.model.nq:])
                gamma_new = -d @ x0[self.model.nq:]

                # print(f"Iteration {r}: gamma = {gamma_new:.4f}, diff = {gamma_new - gamma:.4f}, status = {status}")

                if gamma_new < gamma + self.tol and status == 0:
                    break
                
                gamma = gamma_new

                # Rollout the solution
                x_sol = np.empty((N + n, self.model.nx))
                u_sol = np.empty((N + n, self.model.nu))    # last control is not used
                for i in range(N):
                    x_sol[i] = self.ocp_solver.get(i, 'x')
                    u_sol[i] = self.ocp_solver.get(i, 'u')
                x_sol[N:] = self.ocp_solver.get(N, 'x')
                u_sol[N:] = np.zeros((n, self.model.nu))

                # Reset the initial guess with the previous solution
                self.setGuess(x_sol, u_sol)
                # Increase the horizon
                N += n
                self.resetHorizon(N)
            else:     
                return None, None, None, status
        if status == 0:
            return x_sol, u_sol, N, status
        else:
            return None, None, None, status