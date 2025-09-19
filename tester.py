import numpy as np

class Model:
    def __init__(self):
        self.env_dimensions = np.array([-1, -10, -100, 1, 10, 100])
        self.npos = 3
        self.nori = 3
        self.nv = 3
        self.v_min = np.array([-5, -50, -500])
        self.v_max = np.array([5, 50, 500])

def main():

    i = 0
    model = Model()
    grid = 1

    q_grid = np.arange(model.env_dimensions[i], model.env_dimensions[i+model.npos] +  grid, grid)
    v_grid = np.arange(model.v_min[i], model.v_max[i] + grid, grid)
    box_max_grid = np.empty(len(q_grid)) * np.nan
    box_min_grid = np.empty(len(q_grid)) * np.nan

    for j in range(len(q_grid)):
        box_max_grid[j] = min(model.env_dimensions[i+3], model.env_dimensions[i+3] - q_grid[j])
        box_min_grid[j] = max(model.env_dimensions[i], model.env_dimensions[i] - q_grid[j])

    print(f"q_grid: {q_grid}\nbox_max_grid: {box_max_grid}\nbox_min_grid: {box_min_grid}")

    box_max_grid = np.tile(box_max_grid, len(v_grid))
    box_min_grid = np.tile(box_min_grid, len(v_grid))

    print(f"q_grid: {q_grid}\nbox_max_grid: {box_max_grid}\nbox_min_grid: {box_min_grid}")

    # b_grid = np.hstack((box_min_grid, box_max_grid))

    q, v = np.meshgrid(q_grid, v_grid)
    q_rav, v_rav = q.ravel(), v.ravel()
    n = len(q_rav)

    print(f"q: {q}\nv: {v}\nq_rav: {q_rav}\nv_rav: {v_rav}")

    nbori = 2*model.npos + model.nori
    x_static = np.zeros(nbori + model.nv)
    x = np.repeat(x_static.reshape(1, len(x_static)), n, axis=0)
    
    print(f"x: {x}")

    x[:, i] = box_min_grid
    x[:,model.npos + i] = box_max_grid
    x[:, nbori + i] = v_rav

    print(f"x_fin: {x}")

if __name__ == '__main__':
    main()