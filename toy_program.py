import numpy as np
from casadi import MX, vertcat, horzcat, dot
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Params:
    N: int
    n: int
    dt: float
    rollout: bool
    nq: int
    nv: int
    nu: int
    nx: int

# Define the OCP problem
def setup_ocp(params):
    # Define dimensions
    nq = params.nq  # Number of position states
    nv = params.nv  # Number of velocity states
    nu = params.nu  # Number of control inputs
    nx = params.nx  # Total state dimension

    # Define symbolic variables
    x = MX.sym("x", nx)  # State: [position, velocity]
    u = MX.sym("u", nu)  # Control input
    p = MX.sym("p", nq)  # Parameters

    # Define dynamics
    A = np.vstack((
        np.hstack((np.zeros((nq,nq)), np.eye(nv))),
        np.zeros((nv,nx))
        ))  # Placeholder dynamics matrix
    B = np.vstack((
        np.zeros((nq,nv)),
        np.eye(nv)
    ))  # Placeholder control matrix
    f_expl = A @ x + B @ u + np.array([[0],[0],[0],[-100],[0],[0]])# Linear dynamics for simplicity
    print(f_expl)

    # Create Acados model
    model = AcadosModel()
    model.name = "minimal_ocp"
    model.x = x
    model.u = u
    model.p = p
    model.f_expl_expr = f_expl

    # Create Acados OCP
    ocp = AcadosOcp()
    ocp.model = model

    # Set dimensions
    
    ocp.dims.N = params.N  # Number of shooting nodes

    # Define cost function
    cost_expr = -dot(p, x[nq:])  # Maximize initial velocity
    ocp.cost.cost_type_0 = "EXTERNAL"
    ocp.model.cost_expr_ext_cost_0 = cost_expr
    ocp.parameter_values = np.zeros(nv)


    # Define constraints
    u_min = np.array([0, -200, -200])
    u_max = np.ones(nu) * 200
    q_min = np.array([-5, -10, -5])
    q_max = np.array([2, 10, 5])
    ocp.constraints.lbx_0 = np.zeros(nq)
    ocp.constraints.ubx_0 = np.zeros(nq)
    ocp.constraints.idxbx_0 = np.arange(nq)
    
    ocp.constraints.lbx = q_min
    ocp.constraints.ubx = q_max
    ocp.constraints.idxbx = np.arange(nq)
    
    ocp.constraints.lbx_e = np.zeros(nv)
    ocp.constraints.ubx_e = np.zeros(nv)
    ocp.constraints.idxbx_e = np.arange(nq,nx)
    
    ocp.constraints.lbu = u_min
    ocp.constraints.ubu = u_max
    ocp.constraints.idxbu = np.arange(nu)

    ocp.constraints.C = np.zeros((nv, nx)) 
    ocp.constraints.D  = np.zeros((nv, nu))

    ocp.constraints.lg = np.zeros(nv)
    ocp.constraints.ug = np.zeros(nv)

    # Solver options
    ocp.solver_options.tf = ocp.dims.N*params.dt  # Time horizon
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"

    return ocp

# Solve the OCP
def solve_ocp(params):
    ocp = setup_ocp(params)
    ocp_solver = AcadosOcpSolver(ocp, json_file="minimal_ocp.json")

    nq = params.nq  # Number of position states
    nv = params.nv  # Number of velocity states
    nu = params.nu  # Number of control inputs
    nx = params.nx  # Total state dimension


    # Set initial guess
    x_guess = np.zeros((ocp.dims.N, ocp.model.x.size()[0]))
    u_guess = np.zeros((ocp.dims.N, ocp.model.u.size()[0]))

    # Set parameter values
    p_values = np.array([0.0, 1.0, 0.0])  # Example parameter values
    p_values /= np.linalg.norm(p_values)  # Normalize the parameter vector
    for i in range(ocp.dims.N + 1):  # Set for all shooting nodes (N + 1 includes terminal node)
        ocp_solver.set(i, "p", p_values)

    C = np.zeros((params.nv, params.nx))
    C[:, params.nq:] = np.eye(params.nv)- p_values[:,None] @ p_values[None,:]
    ocp_solver.constraints_set(0, "C", C, api='new')


    # Solve the problem
    
    gamma = 0

    while True:

        ocp_solver.reset()
        for i in range(ocp.dims.N):
            ocp_solver.set(i, "x", x_guess[i])
            ocp_solver.set(i, "u", u_guess[i])

        p_values = np.array([0.0, 1.0, 0.0])  # Example parameter values
        p_values /= np.linalg.norm(p_values)  # Normalize the parameter vector
        for i in range(ocp.dims.N + 1):  # Set for all shooting nodes (N + 1 includes terminal node)
            ocp_solver.set(i, "p", p_values)

        
        status = ocp_solver.solve()

        if status == 0:

            # Extract the state over time
            states = []
            inputs = []
            for i in range(ocp.dims.N):  # Include terminal state
                states.append(ocp_solver.get(i, "x"))
                inputs.append(ocp_solver.get(i, "u"))
            states.append(ocp_solver.get(ocp.dims.N, "x"))  # Terminal state
            inputs.append(np.zeros((nu,)))
            states = np.array(states, dtype=float)
            inputs = np.array(inputs, dtype=float)

            gamma_new = dot(p_values, states[0][nq:])  # Example cost function evaluation

            print(f"gamma_new: {gamma_new}, gamma: {gamma }")

            if gamma_new <= gamma  or params.rollout == 0:
                print("OCP solved successfully with N:", ocp.dims.N)
                
                                # Create time vector
                time = np.linspace(0, ocp.solver_options.tf, ocp.dims.N + 1)

                # Create subplots
                fig, axs = plt.subplots(3, 1, figsize=(10, 8))

                # Plot the first three states (e.g., position states)
                for state_idx in range(nq):  # First three states
                    axs[0].plot(time, states[:, state_idx], label=f"Position State {state_idx}")
                axs[0].set_xlabel("Time [s]")
                axs[0].set_ylabel("Position State Values")
                axs[0].set_title("Position States Evolution Over Time")
                axs[0].legend()
                axs[0].grid()

                # Plot the remaining states (e.g., velocity states)
                for state_idx in range(nq, states.shape[1]):  # Remaining states
                    axs[1].plot(time, states[:, state_idx], label=f"Velocity State {state_idx - 3}")
                axs[1].set_xlabel("Time [s]")
                axs[1].set_ylabel("Velocity State Values")
                axs[1].set_title("Velocity States Evolution Over Time")
                axs[1].legend()
                axs[1].grid()

                # Plot the control inputs
                for input_idx in range(nu): # Control inputs
                    axs[2].plot(time, inputs[:, input_idx], label=f"Control Input {input_idx}")   
                axs[2].set_xlabel("Time [s]")
                axs[2].set_ylabel("Control Input Values")
                axs[2].set_title("Control Inputs Over Time")
                axs[2].legend()
                axs[2].grid()
                
                # Adjust layout and show the plot
                plt.tight_layout()
                plt.show()
                
                break


            else:
                gamma = gamma_new
                x_guess = states
                u_guess = inputs
                # Rollout the solution
                ocp.dims.N += params.n
                ocp_solver.set_new_time_steps(np.full(ocp.dims.N, params.dt))
                ocp_solver.update_qp_solver_cond_N(ocp.dims.N)

        else:
            print(f"OCP solver failed with status {status}")

def main():
    parameters = Params(
        N=10,
        n=1,
        dt=5e-3,
        rollout = 1,
        nq=3,
        nv=3,
        nu=3,
        nx=3+3
    )
    
    solve_ocp(parameters)

if __name__ == "__main__":
    main()

