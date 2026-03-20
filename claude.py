if __name__ == '__main__':
    start_time = time.time()

    # --- Parse command-line arguments ---
    global args, params
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
    params.generation = args['generation']
    params.check = args['check']
    params.build = args['build']
    params.plot = args['plot']
    params.training = args['training']
    params.act = args['activation']
    params.weight_decay = args['weightDecay']

    # --- Initialize model and controller ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global model, controller
    model = Model(params)
    controller = ViabilityController(model)
    nq = model.nq
    nu = model.nu
    if not os.path.exists(params.DATA_DIR):
        os.makedirs(params.DATA_DIR)
    if not os.path.exists(params.NN_DIR):
        os.makedirs(params.NN_DIR)
    N = params.N
    N_increment = params.N_increment
    vboc_repeat = params.vboc_repeat
    horizon = args['horizon']
    plots_dir = params.PLOTS_DIR
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
    act_fun = nls[params.act]
    nn_filename = f'{params.NN_DIR}{robotic_system}_{params.act}.pt'
    ub = 1

    # =========================================================================
    # DATA GENERATION
    # =========================================================================
    if params.generation:

        # In check mode, solve a single problem with a fixed configuration
        if params.check:
            params.prob_num = 1

        # --- Initial position: origin for all problems ---
        pos_init = np.zeros((params.prob_num, model.npos))

        # --- Initial orientation: sampled within the allowed inclination range ---
        if(params.orient_g_rej):
            min_phi = 0.0
            max_phi = model.phi_hovering
        else:
            min_phi = model.phi_max
            max_phi = np.pi/2
        roll, pitch, yaw = generate_constrained_rpy(min_phi, max_phi, params.prob_num)
        if params.check:
            orient_init = np.zeros((params.prob_num, model.nori))
        else:
            orient_init = np.column_stack([roll, pitch, yaw])
        q_init = np.hstack([pos_init, orient_init])

        # --- Obstacle box bounds ---
        box_min_values = np.empty((params.prob_num, model.npos))
        box_max_values = np.empty((params.prob_num, model.npos))
        if params.check:
            # Fixed maximum box in check mode
            for i in range(params.prob_num):
                box_min_values[i,:] = np.array([model.max_width, model.max_length, model.max_height])
                box_max_values[i,:] = np.array([model.max_width, model.max_length, model.max_height])
        else:
            # Random box, with minimum size determined by the drone's ellipsoidal occupancy
            for i in range(params.prob_num):
                min_dx = np.sqrt(np.array([1,0,0]) @ model.Q(np.hstack([q_init[i,:], np.zeros(model.nv)])) @ np.array([1,0,0]).T)
                min_dy = np.sqrt(np.array([0,1,0]) @ model.Q(np.hstack([q_init[i,:], np.zeros(model.nv)])) @ np.array([0,1,0]).T)
                min_dz = np.sqrt(np.array([0,0,1]) @ model.Q(np.hstack([q_init[i,:], np.zeros(model.nv)])) @ np.array([0,0,1]).T)
                dx = np.random.uniform(min_dx, model.max_width)
                dy = np.random.uniform(min_dy, model.max_length)
                dz = np.random.uniform(min_dz, model.max_height)
                box_min_values[i, :] = np.array([dx, dy, dz])
                dx = np.random.uniform(min_dx, model.max_width)
                dy = np.random.uniform(min_dy, model.max_length)
                dz = np.random.uniform(min_dz, model.max_height)
                box_max_values[i, :] = np.array([dx, dy, dz])

        # --- Random seeds, one per problem ---
        randomSeeds = [random.randint(0, params.prob_num) for _ in range(params.prob_num)]

        # --- Accumulators for results across all batches ---
        all_x_0, all_x_t, all_u_t, all_b_m, all_b_M, all_status, all_d_list = [],[],[],[],[],[],[]

        # Split the problems into sub-batches to allow intermediate saves
        if params.check:
            sub_batch = 1
        else:
            sub_batch = 100
        n_batch = int(params.prob_num/sub_batch)

        # --- Per-batch storage (overwritten at each save) ---
        x_data, x_traj, u_traj, b_min, b_max = [], [], [], [], []
        solved = 0
        print('Start data generation')

        for nb in range(n_batch):
            with Pool(params.cpu_num) as p:
                res = p.starmap(computeDataOnBorder, [(q0, N, N_increment, vboc_repeat, box_min, box_max, randomSeeds) for q0, box_min, box_max, randomSeeds in \
                                                      zip(q_init[(nb*sub_batch):((nb+1)*sub_batch)], \
                                                          box_min_values[(nb*sub_batch):((nb+1)*sub_batch)], \
                                                          box_max_values[(nb*sub_batch):((nb+1)*sub_batch)], \
                                                          randomSeeds[(nb*sub_batch):((nb+1)*sub_batch)])])

            # --- Unpack parallel results ---
            x_0, x_t, u_t, b_m, b_M, status, d_list = zip(*res)
            all_x_0.extend(x_0)
            all_x_t.extend(x_t)
            all_u_t.extend(u_t)
            all_b_m.extend(b_m)
            all_b_M.extend(b_M)
            all_status.extend(status)
            all_d_list.extend(d_list)

            # Warn and skip if no feasible solution was found in this batch
            if all(item is None for item in x_0):
                warnings.warn(f'No solution found for any problem in batch {nb}. Skipping this batch.', RuntimeWarning)
                print(status)
                continue
            # Abort if no feasible solution has been found across all batches so far
            if all(item is None for item in all_x_0):
                warnings.warn('No solution found for any problem. Exiting the program.', RuntimeWarning)
                print(status)
                exit()

            # --- Intermediate save: filter out failed problems ---
            x_data = np.vstack([i for i in all_x_0 if i is not None])
            x_traj = [i for i in all_x_t if i is not None]
            u_traj = [i for i in all_u_t if i is not None]
            b_min = list(all_b_m)
            b_max = list(all_b_M)
            d = list(all_d_list)
            status = list(all_status)
            b_combined = np.vstack([np.hstack((b_min[i], b_max[i])) for i in range(len(b_min))])
            np.save(f'{params.DATA_DIR}{robotic_system}_d_vboc', d)
            np.save(f'{params.DATA_DIR}{robotic_system}_b_all_vboc', b_combined)
            np.save(f'{params.DATA_DIR}{robotic_system}_status_vboc', status)
            b_min_succ = [all_b_m[i] for i in range(len(all_b_m)) if all_x_0[i] is not None]
            b_max_succ = [all_b_M[i] for i in range(len(all_b_M)) if all_x_0[i] is not None]
            b_combined_succ = np.vstack([np.hstack((b_min_succ[i], b_max_succ[i])) for i in range(len(b_min_succ))])
            solved = len(x_data)
            print(f'Batch {nb}: Total number of points saved until now: %d' % solved)
            np.save(f'{params.DATA_DIR}{robotic_system}_x_vboc', x_data)
            np.save(f'{params.DATA_DIR}{robotic_system}_b_vboc', b_combined_succ)

        print('Total number of points solved: %d' % solved)

        # --- Plot generated trajectories ---
        if params.plot:

            # Labels and titles for pose/velocity subplots
            extended_pose_title = ['Position', 'Orientation', 'Inclination']
            velocities_title = ['Linear velocity', 'Angular velocity']
            pose_label = ['x [m]', 'y [m]', 'z [m]', 'r [deg]', 'p [deg]', 'y [deg]']
            vel_label = ['v$_x$ [m/s]', 'v$_y$ [m/s]', 'v$_z$ [m/s]', '$\omega_x$ [deg/s]', '$\omega_y$ [deg/s]', '$\omega_z$ [deg/s]']
            y_lab_pose = ['Pos. [m]', 'Orient. [deg]', 'Incl. [deg]']
            y_lab_vel = ['v [m/s]', '$\omega$ [deg/s]']

            # Create (or recreate) output subdirectories
            traj_dir = os.path.join(plots_dir, 'trajectories')
            pose_dir = os.path.join(plots_dir, 'poses')
            velocity_dir = os.path.join(plots_dir, 'velocities')
            input_dir = os.path.join(plots_dir, 'inputs')
            threeD_dir = os.path.join(plots_dir, '3D')
            plots_subdirs = [traj_dir, pose_dir, velocity_dir, input_dir, threeD_dir]
            for subdir in plots_subdirs:
                ensure_clean_dir(subdir)

            # Unit normals used to project the occupancy ellipsoid onto each axis
            normals = [
                np.array([1,0,0]),
                np.array([0,1,0]),
                np.array([0,0,1])
            ]

            # Plot every trajectory, or 1 in 10 outside check mode
            if params.check:
                sub_plot = 1
            else:
                sub_plot = params.prob_num / 10

            for k in range(len(x_traj)):
                if k % sub_plot == 0:
                    horizon_ = x_traj[k].shape[0]
                    colors = np.linspace(0, 1, horizon_)
                    t = np.linspace(0, horizon_ * params.dt, horizon_)
                    traj_xlim_min = (-b_min[k]).tolist() + [-np.rad2deg(max_phi), -np.rad2deg(max_phi), -180.0]
                    traj_xlim_max = b_max[k].tolist() + [np.rad2deg(max_phi), np.rad2deg(max_phi), 180.0]

                    # Phase-plane plot: position vs velocity for each DOF
                    fig, ax = plt.subplots(2, 3)
                    ax = ax.reshape(-1)
                    for i in range(nq):
                        ax[i].grid(True, linewidth=0.5)
                        if i < model.npos:
                            ax[i].scatter(x_traj[k][:, i], x_traj[k][:, nq + i], c=colors, cmap='coolwarm', s=1)
                        else:
                            ax[i].scatter(np.rad2deg(x_traj[k][:, i]), np.rad2deg(x_traj[k][:, nq + i]), c=colors, cmap='coolwarm', s=1)
                        ax[i].set_xlim([traj_xlim_min[i], traj_xlim_max[i]])
                        ax[i].set_xlabel(f'{pose_label[i]}')
                        ax[i].set_ylabel(f'{vel_label[i]}')
                    plt.suptitle(f'Trajectory {k + 1}, d {all_d_list[k][:3]}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(traj_dir, f'traj_{k + 1}.png'))
                    plt.close(fig)

                    # Pose over time, with occupancy ellipsoid bounds for position DOFs
                    fig, ax = plt.subplots(3, 1)
                    ax = ax.reshape(-1)
                    j = 0
                    for i in range(nq):
                        if i == model.npos:
                            j += 1
                        ax[j].grid(True)
                        ax[j].set_title(f'{extended_pose_title[j]}')
                        if i < model.npos:
                            line, = ax[j].plot(t, x_traj[k][:, i], label=f'{pose_label[i]}')
                            ellips_r = []
                            for h in range(len(t)):
                                ellips_r.append(np.sqrt(normals[i].T @ model.Q(x_traj[k][h, :]) @ normals[i]))
                            ax[j].plot(t, x_traj[k][:, i] + ellips_r, color=line.get_color(), linestyle='--', linewidth=0.8)
                            ax[j].plot(t, x_traj[k][:, i] - ellips_r, color=line.get_color(), linestyle='--', linewidth=0.8)
                        else:
                            line, = ax[j].plot(t, np.rad2deg(x_traj[k][:, i]), label=f'{pose_label[i]}')
                        # ax[j].axhline(traj_xlim_min[i], color=line.get_color(), linestyle='--', linewidth=0.8)
                        # ax[j].axhline(traj_xlim_max[i], color=line.get_color(), linestyle='--', linewidth=0.8)
                        ax[j].set_xlabel('Time [s]')
                        ax[j].set_ylabel(y_lab_pose[j])
                        ax[j].legend()
                    j += 1
                    # Last subplot: total inclination angle with safety thresholds
                    ax[j].grid(True)
                    ax[j].set_title(f'{extended_pose_title[j]}')
                    line, = ax[j].plot(t, np.rad2deg(np.sqrt(np.square(x_traj[k][:, 3]) + np.square(x_traj[k][:, 4]))), label=f'{pose_label[i]}')
                    # ax[j].axhline(min_phi, color=line.get_color(), linestyle='--', linewidth=0.8)
                    ax[j].axhline(np.rad2deg(max_phi), color=line.get_color(), linestyle='--', linewidth=0.8)
                    ax[j].axhline(np.rad2deg(model.phi_hovering_max), color='r', linestyle='--', linewidth=0.8)
                    ax[j].set_xlabel('Time [s]')
                    ax[j].set_ylabel(y_lab_pose[j])
                    ax[j].legend()
                    plt.suptitle(f'Trajectory {k + 1}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(pose_dir, f'pose_{k + 1}.png'))
                    plt.close(fig)

                    # Linear and angular velocity over time
                    fig, ax = plt.subplots(2, 1)
                    ax = ax.reshape(-1)
                    j = 0
                    for i in range(nq):
                        if i == model.npos:
                            j += 1
                        ax[j].grid(True)
                        ax[j].set_title(f'{velocities_title[j]}')
                        if i < model.npos:
                            line, = ax[j].plot(t, x_traj[k][:, i + nq], label=f'{vel_label[i]}')
                        else:
                            line, = ax[j].plot(t, np.rad2deg(x_traj[k][:, i + nq]), label=f'{vel_label[i]}')
                        ax[j].set_xlabel('Time [s]')
                        ax[j].set_ylabel(y_lab_vel[j])
                        ax[j].legend()
                    plt.suptitle(f'Trajectory {k + 1}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(velocity_dir, f'vel_{k + 1}.png'))
                    plt.close(fig)

                    # Control inputs over time
                    offset = 200
                    fig, ax = plt.subplots()
                    for i in range(nu):
                        ax.grid(True)
                        ax.plot(t, u_traj[k][:, i], label=f'u_{i + 1}')
                        ax.set_title('Inputs')
                        ax.axhline(model.u_bar, color='r', linestyle='--', lw=0.8)
                        ax.set_xlabel('Time [s]')
                        ax.set_ylabel('$u^2$ [(Hz/s)$^2$]')
                        ax.set_ylim([0.0 - offset, model.u_bar+offset])
                        ax.legend()
                    plt.suptitle(f'Trajectory {k + 1}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(input_dir, f'input_{k + 1}.png'))
                    plt.close(fig)

                    # 3D position trajectory with body-frame axes every 10 steps
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    sc = ax.scatter(x_traj[k][:, 0], x_traj[k][:, 1], x_traj[k][:, 2], c=colors, cmap='coolwarm', s=10)

                    # Overlay body-frame arrows to visualise orientation along the path
                    for i in range(0, len(x_traj[k]), 10):
                        roll, pitch, yaw = x_traj[k][i,3:6]
                        rotation_matrix = Rot.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
                        x_arrow = rotation_matrix[:, 0] * model.min_width
                        y_arrow = rotation_matrix[:, 1] * model.min_length
                        z_arrow = rotation_matrix[:, 2] * model.min_height
                        ax.quiver(
                            x_traj[k][i,0], x_traj[k][i,1], x_traj[k][i,2],
                            x_arrow[0], x_arrow[1], x_arrow[2], color='b', label='X-axis' if i == 0 else ""
                        )
                        ax.quiver(
                            x_traj[k][i,0], x_traj[k][i,1], x_traj[k][i,2],
                            y_arrow[0], y_arrow[1], y_arrow[2], color='r', label='Y-axis' if i == 0 else ""
                        )
                        ax.quiver(
                            x_traj[k][i,0], x_traj[k][i,1], x_traj[k][i,2],
                            z_arrow[0], z_arrow[1], z_arrow[2], color='g', label='Z-axis' if i == 0 else ""
                        )

                    ax.set_xlabel('X [m]')
                    ax.set_ylabel('Y [m]')
                    ax.set_zlabel('Z [m]')
                    ax.set_xlim(traj_xlim_min[0], traj_xlim_max[0])
                    ax.set_ylim(traj_xlim_min[1], traj_xlim_max[1])
                    ax.set_zlim(traj_xlim_min[2], traj_xlim_max[2])
                    ax.set_title(f'3D Position Trajectory {k + 1}')
                    set_axes_equal(ax)
                    # plt.colorbar(sc, ax=ax, label='Time progression')
                    plt.tight_layout()
                    plt.savefig(os.path.join(threeD_dir, f'3D_traj_{k + 1}.png'))
                    plt.close(fig)

    # =========================================================================
    # NEURAL NETWORK TRAINING
    # =========================================================================
    if params.training:

        # --- Load data ---
        x_data = np.load(f'{params.DATA_DIR}{robotic_system}_x_vboc.npy')
        b_data = np.load(f'{params.DATA_DIR}{robotic_system}_b_vboc.npy')
        b_all_data = np.load(params.DATA_DIR + 'sth_b_all_vboc.npy')
        d_data = np.load(params.DATA_DIR + 'sth_d_vboc.npy')
        status_data = np.load(params.DATA_DIR + 'sth_status_vboc.npy')

        # --- Histograms of raw data distributions ---
        if params.plot:
            hist_dir = os.path.join(plots_dir, 'histograms')
            ensure_clean_dir(hist_dir)
            b_all_data = np.load(f'{params.DATA_DIR}{robotic_system}_b_all_vboc.npy')
            d_data = np.load(f'{params.DATA_DIR}{robotic_system}_d_vboc.npy')
            status_data = np.load(f'{params.DATA_DIR}{robotic_system}_status_vboc.npy')
            plot_histogram(x_data[:,:6], title="x[0:6]", xlabel="Value", ylabel="Frequency", bins=50, saving_dir=hist_dir)
            plot_histogram(x_data[:,6:], title="x[6:12]", xlabel="Value", ylabel="Frequency", bins=50, saving_dir=hist_dir)
            plot_histogram(b_data, title="b", xlabel="Value", ylabel="Frequency", bins=50, saving_dir=hist_dir)
            plot_histogram(b_all_data, title="b_all", xlabel="Value", ylabel="Frequency", bins=50, saving_dir=hist_dir)
            plot_histogram(-d_data, title="d", xlabel="Value", ylabel="Frequency", bins=50, saving_dir=hist_dir)
            plot_histogram(status_data, title="status", xlabel="Value", ylabel="Frequency", bins=10, saving_dir=hist_dir)

        # Drop position columns and prepend box dimensions as input features
        x_data = np.hstack((b_data, x_data[:, model.npos:]))

        # --- Shuffle and split into training / validation / test sets ---
        np.random.shuffle(x_data)
        n = len(x_data)
        nbori = model.nbox + model.nori
        train_size = int(params.train_ratio * n)
        val_size = int(params.val_ratio * n)
        test_size = n - train_size - val_size

        x_train = x_data[:train_size]
        x_val = x_data[train_size:train_size + val_size]
        x_test = x_data[train_size + val_size:]

        # --- Standardize box + orientation features using training statistics ---
        mean = np.mean(x_train[:, :nbori])
        std = np.std(x_train[:, :nbori])
        for x_input in [x_train, x_val, x_test]:
            x_input[:, :nbori] = (x_input[:, :nbori] - mean) / std

        # Normalize velocity components by their L2 norm (output = the norm itself)
        y_data = np.linalg.norm(x_data[:, nbori:], axis=1).reshape(n, 1)
        for k in range(n):
            if y_data[k] != 0.:
                x_data[k, nbori:] /= y_data[k]

        # --- Split outputs ---
        y_train = y_data[:train_size]
        y_val = y_data[train_size:train_size + val_size]
        y_test = y_data[train_size + val_size:]

        # --- Build model, loss, and optimiser ---
        nx_train = nbori+model.nv
        nn_model = NeuralNetwork(nx_train, params.hidden_size, 1, params.hidden_layers, act_fun, ub).to(device)
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(nn_model.parameters(),
                                     lr=params.learning_rate,
                                     weight_decay=2e-5,
                                     amsgrad=True)
        regressor = RegressionNN(params, nn_model, loss_fn, optimizer)

        # --- Convert arrays to tensors ---
        x_train = torch.Tensor(x_train).to(device)
        y_train = torch.Tensor(y_train).to(device)
        x_val = torch.Tensor(x_val).to(device)
        y_val = torch.Tensor(y_val).to(device)
        x_test = torch.Tensor(x_test).to(device)
        y_test = torch.Tensor(y_test).to(device)

        # --- Train ---
        print('***START TRAINING***\n')
        train_val_dir = os.path.join(plots_dir, 'training_validation')
        ensure_clean_dir(train_val_dir)
        train_evol, val_evol = regressor.training(x_train, y_train, x_val, y_val, args['epochs'])
        print('***TRAINING COMPLETED***\n')

        # --- Evaluate on training+validation and test sets ---
        print('***MODEL EVALUATION***')
        rmse_train, rel_err = regressor.testing(torch.cat((x_train, x_val), dim=0), torch.cat((y_train, y_val), dim=0))
        print(f'RMSE on Training data: {rmse_train:.5f}')
        print(f'Maximum error wrt training data: {torch.max(torch.abs(rel_err)).item():.5f}')
        rmse_test, rel_err = regressor.testing(x_test, y_test)
        print('---')
        print(f'RMSE on Test data: {rmse_test:.5f}')
        print(f'99 % of the data has a relative error lower than: {torch.quantile(rel_err, 0.99).item():.5f}%')
        print(f'Maximum relative error wrt test data: {torch.max(torch.abs(rel_err)).item():.5f}')
        print('*---*---*---*\n')

        # --- Save model weights and normalisation statistics ---
        torch.save({
            'model': nn_model.state_dict(),
            'mean': mean,
            'std': std,
        }, nn_filename)

        # --- Plot training and validation loss curves ---
        loss_dir = os.path.join(plots_dir, 'loss_evolution')
        ensure_clean_dir(loss_dir)
        fig = plt.figure()
        plt.grid(True, which='both')
        plt.semilogy(train_evol, label='Training', c='b', lw=2)
        plt.semilogy(val_evol, label='Validation', c='g', lw=2)
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss (LP filtered)')
        plt.title(f'Training evolution, horizon {N}')
        plt.savefig(os.path.join(loss_dir, f'evolution_{N}.png'))
        plt.close(fig)

    # =========================================================================
    # VIABILITY KERNEL PLOTTING
    # =========================================================================
    if params.plot and not params.generation:

        # --- Load trained network ---
        device = torch.device("cpu")
        nbori = model.nbox+model.nori
        nx_train = nbori+model.nv
        nn_data = torch.load(nn_filename)
        nn_model = NeuralNetwork(nx_train, params.hidden_size, 1, params.hidden_layers, act_fun, ub).to(device)
        nn_model.load_state_dict(nn_data['model'])
        print('***PLOTTING BRS***\n')

        # Compute fixed-direction section data if not already cached
        if not os.path.exists(f'{params.DATA_DIR}{robotic_system}_x_fixed_vboc.npy'):
            x_fixed, x_status = fixedVelocityDir(N, N_increment, vboc_repeat, n_pts=100)
            np.save(f'{params.DATA_DIR}{robotic_system}_x_fixed_vboc', np.array(x_fixed, dtype=object), allow_pickle=True)
            np.save(f'{params.DATA_DIR}{robotic_system}_status_fixed_vboc', np.array(x_status, dtype=object), allow_pickle=True)
        else:
            x_fixed = np.load(f'{params.DATA_DIR}{robotic_system}_x_fixed_vboc.npy', allow_pickle=True)
            x_status = np.load(f'{params.DATA_DIR}{robotic_system}_status_fixed_vboc.npy', allow_pickle=True)
        brs_dir = os.path.join(plots_dir, 'brs')
        ensure_clean_dir(brs_dir)
        plot_brs(params, model, controller, nn_model, nn_data['mean'], nn_data['std'], x_fixed, x_status)

    print('***ALL DONE***')
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f'Elapsed time: {hours}:{minutes:2d}:{seconds:2d}')
    os.system('aplay /home/maxbertus/Music/notification.wav > /dev/null 2>&1')