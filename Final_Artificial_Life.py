import taichi as ti
import argparse
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import argparse
import pickle

real = ti.f32
ti.init(default_fp=real, arch=ti.cpu, flatten_if=True)

dim = 2
n_particles = 8192
n_solid_particles = 0
n_actuators = 0
n_grid = 128
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 1e-3 # og dt value
#dt = 1e-2
p_vol = 1
E = 10
# TODO: update
mu = E
la = E
max_steps = 2048
steps = 2048
# These are the og values below
# max_steps = 2048
# steps = 2048
#steps = 1024 of value
gravity = 2.4 #1 for sure works.
target = [0.8, 0.2]
params = {}

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(dim, dtype=real)
mat = lambda: ti.Matrix.field(dim, dim, dtype=real)

actuator_id = ti.field(ti.i32)
particle_type = ti.field(ti.i32)
x, v = vec(), vec()
grid_v_in, grid_m_in = vec(), scalar()
grid_v_out = vec()
C, F = mat(), mat()

loss = scalar()

n_sin_waves = 4
weights = scalar()
bias = scalar()
x_avg = vec()
y_avg = vec() #adding this to show how high. 

actuation = scalar()
actuation_omega = 20
act_strength = 4



def allocate_fields():
    ti.root.dense(ti.ij, (n_actuators, n_sin_waves)).place(weights)
    ti.root.dense(ti.i, n_actuators).place(bias)

    ti.root.dense(ti.ij, (max_steps, n_actuators)).place(actuation)
    ti.root.dense(ti.i, n_particles).place(actuator_id, particle_type)
    ti.root.dense(ti.k, max_steps).dense(ti.l, n_particles).place(x, v, C, F)
    ti.root.dense(ti.ij, n_grid).place(grid_v_in, grid_m_in, grid_v_out)
    ti.root.place(loss, x_avg, y_avg)

    ti.root.lazy_grad()


@ti.kernel
def clear_grid():
    for i, j in grid_m_in:
        grid_v_in[i, j] = [0, 0]
        grid_m_in[i, j] = 0
        grid_v_in.grad[i, j] = [0, 0]
        grid_m_in.grad[i, j] = 0
        grid_v_out.grad[i, j] = [0, 0]


@ti.kernel
def clear_particle_grad():
    # for all time steps and all particles
    for f, i in x:
        x.grad[f, i] = [0, 0]
        v.grad[f, i] = [0, 0]
        C.grad[f, i] = [[0, 0], [0, 0]]
        F.grad[f, i] = [[0, 0], [0, 0]]


@ti.kernel
def clear_actuation_grad():
    for t, i in actuation:
        actuation[t, i] = 0.0


@ti.kernel
def p2g(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_F = (ti.Matrix.diag(dim=2, val=1) + dt * C[f, p]) @ F[f, p]
        J = (new_F).determinant()
        if particle_type[p] == 0:  # fluid
            sqrtJ = ti.sqrt(J)
            new_F = ti.Matrix([[sqrtJ, 0], [0, sqrtJ]])

        F[f + 1, p] = new_F
        r, s = ti.polar_decompose(new_F)

        act_id = actuator_id[p]

        act = actuation[f, ti.max(0, act_id)] * act_strength
        if act_id == -1:
            act = 0.0
        # ti.print(act)

        A = ti.Matrix([[0.0, 0.0], [0.0, 1.0]]) * act
        cauchy = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        mass = 0.0
        if particle_type[p] == 0:
            mass = 4
            cauchy = ti.Matrix([[1.0, 0.0], [0.0, 0.1]]) * (J - 1) * E
        else:
            mass = 1
            cauchy = 2 * mu * (new_F - r) @ new_F.transpose() + \
                     ti.Matrix.diag(2, la * (J - 1) * J)
        cauchy += new_F @ A @ new_F.transpose()
        stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
        affine = stress + mass * C[f, p]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (ti.cast(ti.Vector([i, j]), real) - fx) * dx
                weight = w[i][0] * w[j][1]
                grid_v_in[base +
                          offset] += weight * (mass * v[f, p] + affine @ dpos)
                grid_m_in[base + offset] += weight * mass


bound = 3
coeff = 0.5


@ti.kernel
def grid_op():
    for i, j in grid_m_in:
        inv_m = 1 / (grid_m_in[i, j] + 1e-10)
        v_out = inv_m * grid_v_in[i, j]
        v_out[1] -= dt * gravity
        if i < bound and v_out[0] < 0:
            v_out[0] = 0
            v_out[1] = 0
        if i > n_grid - bound and v_out[0] > 0:
            v_out[0] = 0
            v_out[1] = 0
        if j < bound and v_out[1] < 0:
            v_out[0] = 0
            v_out[1] = 0
            normal = ti.Vector([0.0, 1.0])
            lsq = (normal**2).sum()
            if lsq > 0.5:
                if ti.static(coeff < 0):
                    v_out[0] = 0
                    v_out[1] = 0
                else:
                    lin = v_out.dot(normal)
                    if lin < 0:
                        vit = v_out - lin * normal
                        lit = vit.norm() + 1e-10
                        if lit + coeff * lin <= 0:
                            v_out[0] = 0
                            v_out[1] = 0
                        else:
                            v_out = (1 + coeff * lin / lit) * vit
        if j > n_grid - bound and v_out[1] > 0:
            v_out[0] = 0
            v_out[1] = 0

        grid_v_out[i, j] = v_out


@ti.kernel
def g2p(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, real)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector([0.0, 0.0])
        new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                dpos = ti.cast(ti.Vector([i, j]), real) - fx
                g_v = grid_v_out[base[0] + i, base[1] + j]
                weight = w[i][0] * w[j][1]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx

        v[f + 1, p] = new_v
        x[f + 1, p] = x[f, p] + dt * v[f + 1, p]
        C[f + 1, p] = new_C


@ti.kernel
def compute_actuation(t: ti.i32):
    for i in range(n_actuators):
        act = 0.0
        for j in ti.static(range(n_sin_waves)):
            act += weights[i, j] * ti.sin(actuation_omega * t * dt +
                                          2 * math.pi / n_sin_waves * j)
        act += bias[i]
        actuation[t, i] = ti.tanh(act)

@ti.kernel
def compute_x_avg():
    for i in range(n_particles):
        contrib = 0.0
        if particle_type[i] == 1:
            contrib = 1.0 / n_solid_particles
        ti.atomic_add(x_avg[None], contrib * x[steps - 1, i])

@ti.kernel
def compute_loss():
    x_dist = x_avg[None][0]
    y_dist = x_avg[None][1]
    loss[None] = -x_dist - y_dist

@ti.ad.grad_replaced
def advance(s):
    clear_grid()
    compute_actuation(s)
    p2g(s)
    grid_op()
    g2p(s)

@ti.ad.grad_for(advance)
def advance_grad(s):
    clear_grid()
    p2g(s)
    grid_op()

    g2p.grad(s)
    grid_op.grad()
    p2g.grad(s)
    compute_actuation.grad(s)

def forward(total_steps=steps):
    # simulation
    for s in range(total_steps - 1):
        advance(s)
    x_avg[None] = [0, 0]
    #y_avg[None] = 0
    compute_x_avg()
    compute_loss()


class Scene:
    def __init__(self):
        self.n_particles = 0
        self.n_solid_particles = 0
        self.x = []
        self.actuator_id = []
        self.particle_type = []
        self.offset_x = 0
        self.offset_y = 0

    def add_rect(self, x, y, w, h, actuation, ptype=1):
        if ptype == 0:
            assert actuation == -1
        global n_particles
        w_count = max(1,int(w / dx) * 2)
        h_count = max(1,int(h / dx) * 2)
        real_dx = w / w_count
        real_dy = h / h_count
        for i in range(w_count):
            for j in range(h_count):
                self.x.append([
                    x + (i + 0.5) * real_dx + self.offset_x,
                    y + (j + 0.5) * real_dy + self.offset_y
                ])
                self.actuator_id.append(actuation)
                self.particle_type.append(ptype)
                self.n_particles += 1
                self.n_solid_particles += int(ptype == 1)

    def add_rotated_rect(self, x, y, w, h,actuation, rotation = 0, ptype=1):
        if ptype == 0:
            assert actuation == -1
        global n_particles
        w_count = max(1,int(w / dx) * 2)
        h_count = max(1,int(h / dx) * 2)
        real_dx = w / w_count
        real_dy = h / h_count

        cos_theta = math.cos(rotation)
        sin_theta = math.sin(rotation)

        for i in range(w_count):
            for j in range(h_count):
                #local coordinates within the rectangle
                local_x = (i + 0.5)*real_dx - w/2
                local_y = (j + 0.5)*real_dy - h/2

                #rotate local coordinates around the center of the rectangle
                rotated_x = local_x * cos_theta - local_y * sin_theta
                rotated_y = local_x * sin_theta + local_y * cos_theta


                #translate rotated coordinates to global position
                global_x = x + rotated_x + self.offset_x
                global_y = y + rotated_y + self.offset_y
                
                # Add the particle
                self.x.append([global_x, global_y])
                self.actuator_id.append(actuation)
                self.particle_type.append(ptype)
                self.n_particles += 1
                self.n_solid_particles += int(ptype == 1)
        

    def add_circle(self, x, y, radius, actuation, ptype=1):
        if ptype == 0:
            assert actuation == -1
        global n_particles
        radius_count = int(radius / dx)*2  #number of particles based on radius
        real_dx = radius / radius_count

        for i in range(-radius_count, radius_count + 1):  #iterate over a square region
            for j in range(-radius_count, radius_count + 1):
                #calculate the position of the particle
                px = x + i * real_dx + self.offset_x
                py = y + j * real_dx + self.offset_y
                
                #check if the particle is within the circle's radius
                if (px - x) ** 2 + (py - y) ** 2 <= radius ** 2:
                    self.x.append([px, py])
                    self.actuator_id.append(actuation)
                    self.particle_type.append(ptype)
                    self.n_particles += 1
                    self.n_solid_particles += int(ptype == 1)

    def add_semicircle(self, x, y, radius, actuation,rotation =0, ptype=1):
        if ptype == 0:
            assert actuation == -1
        global n_particles
        radius_count = int(radius / dx)*2  #mumber of particles based on radius
        real_dx = radius / radius_count
        for i in range(-radius_count, radius_count + 1):  #iterate over a square region
            for j in range(0, radius_count + 1):  #only the upper half (y >= 0)
                #local coordinates within the semicircle
                local_x = i * real_dx
                local_y = j * real_dx

                #check if the point lies within the semicircle
                if local_x**2 + local_y**2 <= radius**2:
                    #apply rotation
                    rotated_x = (
                        local_x * math.cos(rotation) - local_y * math.sin(rotation)
                    )
                    rotated_y = (
                        local_x * math.sin(rotation) + local_y * math.cos(rotation)
                    )

                    #global coordinates
                    px = rotated_x + x + self.offset_x
                    py = rotated_y + y + self.offset_y

                    self.x.append([px, py])
                    self.actuator_id.append(actuation)
                    self.particle_type.append(ptype)
                    self.n_particles += 1
                    self.n_solid_particles += int(ptype == 1)



    def set_offset(self, x, y):
        self.offset_x = x
        self.offset_y = y

    def finalize(self):
        global n_particles, n_solid_particles
        n_particles = self.n_particles
        n_solid_particles = self.n_solid_particles
        print('n_particles', n_particles)
        print('n_solid', n_solid_particles)

    def set_n_actuators(self, n_act):
        global n_actuators
        n_actuators = n_act



def save_checkpoint(gen, population, generation_loss_histories, checkpoint_file='checkpoint.pkl'):
    checkpoint_data = {
        'generation': gen,
        'population': population,
        'generation_loss_histories': generation_loss_histories
    }
    
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print(f"Checkpoint saved at generation {gen}.")

def load_checkpoint(checkpoint_file='checkpoint.pkl'):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
        print(f"Checkpoint loaded: Generation {checkpoint_data['generation']}.")
        return checkpoint_data
    else:
        print("No checkpoint found, starting from scratch.")
        return None



def fish(scene):
    scene.add_rect(0.025, 0.025, 0.95, 0.1, -1, ptype=0)
    scene.add_rect(0.1, 0.2, 0.15, 0.05, -1)
    scene.add_rect(0.1, 0.15, 0.025, 0.05, 0)
    scene.add_rect(0.125, 0.15, 0.025, 0.05, 1)
    scene.add_rect(0.2, 0.15, 0.025, 0.05, 2)
    scene.add_rect(0.225, 0.15, 0.025, 0.05, 3)
    scene.set_n_actuators(4)

def robot(scene):
    scene.set_offset(0.1, 0.03)
    scene.add_rect(0.1, 0.2, 0.15, 0.05, -1)
    scene.add_rect(0.1, 0.15, 0.025, 0.05, 0)
    scene.add_rect(0.125, 0.15, 0.025, 0.05, 1)
    scene.add_rect(0.2, 0.15, 0.025, 0.05, 2)
    scene.add_rect(0.225, 0.15, 0.025, 0.05, 3)
    scene.set_n_actuators(4)


def cell_generator(scene, params=None):
    if params is None:
        params = {
            'body_radius': random.uniform(0.008, 0.05),
            'pad_radius': random.uniform(0.008, 0.01),
            'num_arms': random.randint(1, 10),
            'arm_length': random.uniform(0.02, 0.05),
            'arm_width': random.uniform(0.04, 0.05),
            'weld_length': random.uniform(0, 0.8 * random.uniform(0.008, 0.01)),  # Adjusted to avoid missing var
            'muscle_count': random.randint(1, 4),
        }

    print(f"Using parameters: {params}")  # Debugging

    body_radius = params['body_radius']
    pad_radius = params['pad_radius']
    num_arms = params['num_arms']
    arm_length = params['arm_length']
    arm_width = params['arm_width']
    weld_length = params['weld_length']
    muscle_count = params['muscle_count']

    max_reach = body_radius + arm_length + pad_radius + 0.05
    x_center = max_reach
    y_center = max_reach

    scene.add_circle(x_center, y_center, body_radius, -1)  # Central body

    angle_step = 2 * math.pi / num_arms
    w_step = arm_length / muscle_count
    h_step = arm_width / muscle_count

    print("w_step:", w_step, "h_step:", h_step)  # Debugging

    for i in range(num_arms):
        angle = i * angle_step
        arm_x = x_center + math.cos(angle) * body_radius
        arm_y = y_center + math.sin(angle) * body_radius
        semi_angle = angle - math.pi / 2

        scene.add_semicircle(
            arm_x + math.cos(angle) * arm_length / 2,
            arm_y + math.sin(angle) * arm_length / 2,
            pad_radius, i, rotation=semi_angle
        )

        for z in range(muscle_count):
            j = i * muscle_count + z
            scene.add_rotated_rect(arm_x, arm_y, w_step, h_step, j, rotation=angle)

            arm_x += math.cos(angle) * (w_step - weld_length)
            arm_y += math.sin(angle) * (h_step - weld_length)

    scene.set_n_actuators(num_arms * muscle_count)
    return params

def mutate_params(params):
    mutated = params.copy()
    keys = ['body_radius', 'pad_radius', 'num_arms', 'arm_length', 'arm_width']
    
    # Choose 3 distinct keys randomly
    selected_keys = random.sample(keys, 3)
    
    for key in selected_keys:
        if key in ['num_arms']:
            # Mutate integer parameters
            mutated[key] += random.choice([-3, -2, -1, 1, 2, 3])
            mutated[key] = max(1, mutated[key])
            mutated[key] = min(3, mutated[key])
        else:
            # Mutate continuous parameters by adding a small random change
            mutation = random.gauss(0, 0.005)  # Small change centered at 0
            mutated[key] += mutation
            mutated[key] = abs(mutated[key])  # Ensure positive values
            mutated[key] = max(0.008, mutated[key])
    return mutated


gui = ti.GUI("Differentiable MPM", (640, 640), background_color=0xFFFFFF)


def visualize(s, folder):
    aid = actuator_id.to_numpy()
    colors = np.empty(shape=n_particles, dtype=np.uint32)
    particles = x.to_numpy()[s]
    actuation_ = actuation.to_numpy()
    for i in range(n_particles):
        color = 0x111111
        if aid[i] != -1:
            act = actuation_[s - 1, int(aid[i])]
            color = ti.rgb_to_hex((0.5 - act, 0.5 - abs(act), 0.5 + act))
        colors[i] = color
    gui.circles(pos=particles, color=colors, radius=1.5)
    gui.line((0.05, 0.02), (0.95, 0.02), radius=3, color=0x0)

    os.makedirs(folder, exist_ok=True)
    gui.show(f'{folder}/{s:04d}.png')


def init_taichi_fields():
    global actuator_id, particle_type, x, v, grid_v_in, grid_m_in, grid_v_out, C, F
    global loss, weights, bias, x_avg, y_avg, actuation

    actuator_id = ti.field(ti.i32)
    particle_type = ti.field(ti.i32)
    x, v = vec(), vec()
    grid_v_in, grid_m_in = vec(), scalar()
    grid_v_out = vec()
    C, F = mat(), mat()

    loss = scalar()
    weights = scalar()
    bias = scalar()
    x_avg = vec()
    y_avg = vec()
    actuation = scalar()


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--iters', type=int, default=20) #number of iterations
#     parser.add_argument('--gens', type=int, default=10)  # number of generations
#     options = parser.parse_args()

#     population_size = 6
#     num_generations = options.gens

#     # Here we create initial population: a list of parameter dictionaries.
#     population = []
#     for _ in range(population_size):
#         temp_scene = Scene() # create a new scene for a new cell
#         params = cell_generator(temp_scene) # return a dictionary of params
#         population.append(params)
    
#     # we need to store loss histories for visualization across generations
#     generation_loss_histories = []

#     for gen in range(num_generations):
#         print(f"\n=== Generation {gen} ===")
#         cells_data = []  # store parameters and metrics for each cell in this generation
#         all_loss_histories = []  # loss history for each cell in this generation

#         # evaluate each cell in the current population:
#         for cell in range(population_size):
#             ti.reset()
#             ti.init(default_fp=ti.f32, arch=ti.cpu, flatten_if=True)
#             init_taichi_fields()
#             print(f"\n--- Running simulation for cell {cell} in generation {gen} ---")
            
#             scene = Scene()
#             # use the parameters from the population
#             params = population[cell]
#             params = cell_generator(scene, params)
#             scene.finalize()
#             allocate_fields()

#             # initialize weights and particle fields as before:
#             for i in range(n_actuators):
#                 for j in range(n_sin_waves):
#                     weights[i, j] = np.random.randn() * 0.01

#             for i in range(scene.n_particles):
#                 x[0, i] = scene.x[i]
#                 F[0, i] = [[1, 0], [0, 1]]
#                 actuator_id[i] = scene.actuator_id[i]
#                 particle_type[i] = scene.particle_type[i]

#             losses = []
#             # run simulation iterations for this cell:
#             for iter in range(options.iters):
#                 with ti.ad.Tape(loss):
#                     forward()
#                 l = loss[None]
#                 losses.append(l)
#                 print(f"cell {cell} | Iteration {iter} | Loss: {l}")
#                 learning_rate = 0.2
#                 #try learning_rate = 0.1
#                 #learning_rate = 0.1
#                 for i in range(n_actuators):
#                     for j in range(n_sin_waves):
#                         weights[i, j] -= learning_rate * weights.grad[i, j]
#                     bias[i] -= learning_rate * bias.grad[i]

#                 # if iter == 0 or iter == options.iters - 1:  # First or last iteration
#                 #     forward(1500)
#                 #     for s in range(15, 1500, 16):
#                 #         visualize(s, f'diffmpm/gen{gen}_cell{cell:03d}/iter{iter:03d}/')


#             # record final metrics:
#             params['final_loss'] = losses[-1]
#             params['improvement'] = losses[0] - losses[-1]  # change in loss over iterations
#             cells_data.append(params)
#             all_loss_histories.append(losses)
        
#         generation_loss_histories.append(all_loss_histories)

#         # print out the parameters and loss for each cell in this generation:
#         for i, data in enumerate(cells_data):
#             print(f"Generation {gen} - Cell {i}: {data}")

#         # use pandas to perform correlation analysis for this generation:
#         df = pd.DataFrame(cells_data)
#         print(f"\nGeneration {gen} Correlation matrix:")
#         print(df.corr())

#         # We sort the generation
#         # lower final_loss (more negative) is better, and higher improvement is better.
#         # sort by final loss (lowest first) and improvement (highest first)
#         sorted_by_performance = sorted(cells_data, key=lambda d: d['final_loss'])
#         sorted_by_improvement = sorted(cells_data, key=lambda d: d['improvement'], reverse=True)

#         # take the top 2 from each category.
#         survivors = sorted_by_performance[:2] + sorted_by_improvement[:2]

#         # remove duplicates (if any)
#         survivors = {id(d): d for d in survivors}.values()
#         survivors = list(survivors)
#         print(f"Generation {gen} Survivors (selected):")
#         for s in survivors:
#             print(s)

#         # we create the new generation
#         new_population = survivors[:]  # start with survivors
#         # ror the remaining slots, mutate a randomly selected cell (for example, from the survivors or entire current generation)
#         while len(new_population) < population_size:
#             candidate = random.choice(cells_data)
#             mutated = mutate_params(candidate)
#             new_population.append(mutated)
#         population = new_population  # set new generation's population

#         #plot the loss histories for this generation:
#         for c in range(population_size):
#             plt.plot(all_loss_histories[c], label=f"cell {c}")
#         plt.title(f"Loss History - Generation {gen}")
#         plt.xlabel("Iteration")
#         plt.ylabel("Loss")
#         plt.legend()
#         plt.show()

#     # analyze the population across generations
#     # print out final parameters of the last generation
#     print("\n=== Final Generation Population ===")
#     for i, params in enumerate(population):
#         print(f"Cell {i}: {params}")


# if __name__ == '__main__':
#     main()



# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--iters', type=int, default=20)  # Number of iterations
#     parser.add_argument('--gens', type=int, default=10)   # Number of generations
#     options = parser.parse_args()

#     population_size = 6
#     num_generations = options.gens

#     # Create initial population
#     population = []
#     for _ in range(population_size):
#         temp_scene = Scene()
#         params = cell_generator(temp_scene)
#         population.append(params)

#     generation_loss_histories = []

#     for gen in range(num_generations):
#         print(f"\n=== Generation {gen} ===")
#         gen_folder = f'diffmpm/gen{gen:02d}/'
#         os.makedirs(gen_folder, exist_ok=True)

#         cells_data = []
#         all_loss_histories = []

#         for cell in range(population_size):
#             ti.reset()
#             ti.init(default_fp=ti.f32, arch=ti.cpu, flatten_if=True)
#             init_taichi_fields()
#             print(f"\n--- Running simulation for cell {cell} in generation {gen} ---")

#             scene = Scene()
#             params = population[cell]
#             params = cell_generator(scene, params)
#             scene.finalize()
#             allocate_fields()

#             for i in range(n_actuators):
#                 for j in range(n_sin_waves):
#                     weights[i, j] = np.random.randn() * 0.01

#             for i in range(scene.n_particles):
#                 x[0, i] = scene.x[i]
#                 F[0, i] = [[1, 0], [0, 1]]
#                 actuator_id[i] = scene.actuator_id[i]
#                 particle_type[i] = scene.particle_type[i]

#             losses = []
#             cell_folder = f"{gen_folder}/cell{cell:03d}/"
#             os.makedirs(cell_folder, exist_ok=True)

#             # Run simulation iterations
#             for iter in range(options.iters):
#                 with ti.ad.Tape(loss):
#                     forward()

#                 l = loss[None]
#                 losses.append(l)
#                 print(f"cell {cell} | Iteration {iter} | Loss: {l}")

#                 learning_rate = 0.2
#                 for i in range(n_actuators):
#                     for j in range(n_sin_waves):
#                         weights[i, j] -= learning_rate * weights.grad[i, j]
#                     bias[i] -= learning_rate * bias.grad[i]

#                 # Save frames at first and last iteration
#                 if iter == 0 or iter == options.iters - 1:
#                     iter_folder = f"{cell_folder}/iter{iter:03d}/"
#                     os.makedirs(iter_folder, exist_ok=True)
#                     forward(2048)  # Run a forward pass
#                     for s in range(0, 2048, 16):  # Save frames every 16 steps
#                         visualize(s, f"{iter_folder}/frame{s:04d}.png")

#             # Record final metrics
#             params['final_loss'] = losses[-1]
#             params['improvement'] = losses[0] - losses[-1]
#             cells_data.append(params)
#             all_loss_histories.append(losses)

#             # Save loss history to CSV
#             loss_csv_path = f"{cell_folder}/loss.csv"
#             pd.DataFrame({"Iteration": range(options.iters), "Loss": losses}).to_csv(loss_csv_path, index=False)

#         generation_loss_histories.append(all_loss_histories)

#         # Sorting and selecting the next generation
#         sorted_by_performance = sorted(cells_data, key=lambda d: d['final_loss'])
#         sorted_by_improvement = sorted(cells_data, key=lambda d: d['improvement'], reverse=True)

#         survivors = sorted_by_performance[:2] + sorted_by_improvement[:2]
#         survivors = {id(d): d for d in survivors}.values()
#         survivors = list(survivors)

#         new_population = survivors[:]
#         while len(new_population) < population_size:
#             candidate = random.choice(cells_data)
#             mutated = mutate_params(candidate)
#             new_population.append(mutated)
#         population = new_population

#         # Plot loss histories for the generation
#         plt.figure(figsize=(8, 5))
#         for c in range(population_size):
#             plt.plot(all_loss_histories[c], label=f"cell {c}")
#         plt.title(f"Loss History - Generation {gen}")
#         plt.xlabel("Iteration")
#         plt.ylabel("Loss")
#         plt.legend()
#         plt.savefig(f"{gen_folder}/loss_history.png")
#         plt.close()

#     print("\n=== Final Generation Population ===")
#     for i, params in enumerate(population):
#         print(f"Cell {i}: {params}")

# if __name__ == '__main__':
#     main()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=50)  # Number of iterations
    parser.add_argument('--gens', type=int, default=10)   # Number of generations
    options = parser.parse_args()

    population_size = 6
    num_generations = options.gens

    # Check for a saved checkpoint
    checkpoint_data = load_checkpoint()
    
    # If there's no checkpoint, create a new population
    if checkpoint_data is None:
        population = []
        for _ in range(population_size):
            temp_scene = Scene()
            params = cell_generator(temp_scene)
            population.append(params)
        generation_loss_histories = []
        start_gen = 0
    else:
        population = checkpoint_data['population']
        generation_loss_histories = checkpoint_data['generation_loss_histories']
        start_gen = checkpoint_data['generation'] + 1  # Start from the next generation
    
    for gen in range(start_gen, num_generations):
        print(f"\n=== Generation {gen} ===")
        gen_folder = f'diffmpm/gen{gen:02d}/'
        os.makedirs(gen_folder, exist_ok=True)

        cells_data = []
        all_loss_histories = []

        for cell in range(population_size):
            ti.reset()
            ti.init(default_fp=ti.f32, arch=ti.cpu, flatten_if=True)
            init_taichi_fields()
            print(f"\n--- Running simulation for cell {cell} in generation {gen} ---")

            scene = Scene()
            params = population[cell]
            params = cell_generator(scene, params)
            scene.finalize()
            allocate_fields()

            for i in range(n_actuators):
                for j in range(n_sin_waves):
                    weights[i, j] = np.random.randn() * 0.01

            for i in range(scene.n_particles):
                x[0, i] = scene.x[i]
                F[0, i] = [[1, 0], [0, 1]]
                actuator_id[i] = scene.actuator_id[i]
                particle_type[i] = scene.particle_type[i]

            losses = []
            cell_folder = f"{gen_folder}/cell{cell:03d}/"
            os.makedirs(cell_folder, exist_ok=True)

            # Run simulation iterations
            for iter in range(options.iters):
                with ti.ad.Tape(loss):
                    forward()

                l = loss[None]
                losses.append(l)
                print(f"cell {cell} | Iteration {iter} | Loss: {l}")

                learning_rate = 0.15
                for i in range(n_actuators):
                    for j in range(n_sin_waves):
                        weights[i, j] -= learning_rate * weights.grad[i, j]
                    bias[i] -= learning_rate * bias.grad[i]

                # Save frames at first and last iteration
                if iter == 0 or iter == options.iters - 1:
                    iter_folder = f"{cell_folder}/iter{iter:03d}/"
                    os.makedirs(iter_folder, exist_ok=True)
                    forward(2048)  # Run a forward pass
                    for s in range(0, 2048, 16):  # Save frames every 16 steps
                        visualize(s, f"{iter_folder}/frame{s:04d}.png")

            # Record final metrics for the cell
            params['final_loss'] = losses[-1]
            params['improvement'] = losses[0] - losses[-1]
            cells_data.append(params)
            all_loss_histories.append(losses)

            # Save loss history to CSV
            loss_csv_path = f"{cell_folder}/loss.csv"
            pd.DataFrame({"Iteration": range(options.iters), "Loss": losses}).to_csv(loss_csv_path, index=False)

        generation_loss_histories.append(all_loss_histories)

        # --- CORRELATION ANALYSIS ---
        # Create a DataFrame from cells_data and compute the correlation matrix.
        df = pd.DataFrame(cells_data)
        corr_matrix = df.corr()
        print(f"\nGeneration {gen} Correlation matrix:")
        print(corr_matrix)
        # Save the correlation matrix as CSV in the generation folder.
        corr_csv_path = os.path.join(gen_folder, "correlation_matrix.csv")
        corr_matrix.to_csv(corr_csv_path)
        # --- END CORRELATION ANALYSIS ---

        # Plot loss histories for the generation
        plt.figure(figsize=(8, 5))
        for c in range(population_size):
            plt.plot(all_loss_histories[c], label=f"cell {c}")
        plt.title(f"Loss History - Generation {gen}")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{gen_folder}/loss_history.png")
        plt.close()

        # Save checkpoint after each generation
        save_checkpoint(gen, population, generation_loss_histories)

    print("\n=== Final Generation Population ===")
    for i, params in enumerate(population):
        print(f"Cell {i}: {params}")

if __name__ == '__main__':
    main()

