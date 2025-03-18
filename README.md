
# DiffTaichi: Differentiable Programming for Physical Simulation

*Damian Gonzalez

By using the Difftaichi software, we are able to run differentiable physics software.
This software has enabled us to create artifical organisms that are able to learn by utilizing neural networks with a brute-force gradient descent method.

This code uses a difftaichi example and modifies it to create randomized cells. These cells are varied with different parameters that alter their arm size, body size, number of arms, size of their pads, and number of muscles.
The code is designated as cell_generator(scene)

There is another function that allows you to also input in values for these parameters. That function is known as add_virus(scene). Follow the limits that are set up in the cell_generator to avoid any errors. 

In order to use this code first you would have to download Difftaichi, below is the steps to download it. 

## How to run
Install [`Taichi`](https://github.com/taichi-dev/taichi) with `pip`:

(Most examples do **not** need a GPU to run.)
```bash
python3 -m pip install taichi
```

Then you can run the script in the Final_Artificial_Life folder known as Final_Artificial_Life.py

Use the folowing function to simulate it.

[`python Final_Artificial_Life.py`]

# Artificial Life Simulation


## Specific Portions in how it is ran

The following information shows the generation of the cells along with the algorithm. 

Circle and Semicircle Shape Generation
I took a similar approach to the add_rect function in that I started off the code with figuring out my radius_count by taking the radius, dividing it by dx as determined in the initialization of my code. The code determines dx by taking the reciprocal of the number that the grid is discretized to. Then we can figure out our real_dx by taking the radius and dividing it by radius_count. We then figure out what particles are going to make up this circle. So, we iterate over a square like figure (by using two for loops that iterate from the negative radius_count to the positive radius count plus one. We do this because it allows us to essentially start from the center of the circle (including the offset) and create this square. After we create this square, we use an if statement to figure out if the point in our square region is within our circle. I used the equation of a circle to check if the point was indeed part of our defined circle. The equation was the following: px-x2+py-y2r2 where x and y are the center of the circle, r is the radius, p_x and p_y are the points we generated. We generated those points through the following equation. 
p_x=x+i*real_dx+self.offset_x  
p_y=x+i*real_dx+self.offset_y  
    If our generated point is within the region defined by the equation of the circle, then we add that point to our final circle. We replicated this same process for creating a semicircle. The only modification that I did to achieve the semicircle is that while I was generating points from -radius_count to positive radius_count plus one, I iterated j over a 0 to radius_count + one. What this did was only account for the top half of the square that was generated in the circle, and thus we were able to create a semicircle. A problem arose however in that these circles were all made with the flat side always facing downward. In making my cell, I wanted to have these tails be protruding outward radially, thus I would need to change the orientation of these semicircles and rectangles to make these tails. To achieve the altered orientation, I added in a point in the code for the semicircles that have a rotation. To do this I modified the code in the following manner. Instead of computing points p_x and p_y, the way I did with the circle, I added a rotation to those points in the following manner. 
Rotated Rectangle Generation
i*real_dx and self.offset_y describe the local coordinates of the square that we are iterating over to obtain our semicircle. We use the same if statement to see if it is within our equation of the circle, but we also add that if it does fit within our semicircle, it will then apply a rotation through the following manner. 
local_x = i*real_dx   
local_y = i*real_dx
rotate_dx=local_x*cos⁡(α)-local_y*sin(α)
rotate_dy=local_x*sin⁡(α)+local_y*cos(α)
After figuring out the rotated coordinates, we can then perform the following to achieve the coordinates for the particle of the semicircle. 
p_x=x+rotate_dx+self.offset_x  
p_y=x+rotate_dy+self.offset_y
Cell Generation


## Initialization of Parameters

First, we initialize the parameters through python’s random function. The following shows how we generated the values. It should be noted that for 〖num〗_arms and 〖muscle〗_count must be randomized integers. It must be because there shouldn’t be any fractional values for number of muscles and arms. 

'body_radius': random.uniform(0.008, 0.05),
'pad_radius': random.uniform(0.008, 0.01),
'num_arms': random.randint(1, 10),
'arm_length': random.uniform(0.02, 0.05),
'arm_width': random.uniform(0.04, 0.05),
'weld_length': random.uniform(0, 0.8 * random.uniform(0.008, 0.01)),  # Adjusted to avoid missing var
'muscle_count': random.randint(1, 4),

The first number represents the minimum number possible for there to be, and the second number represents the highest number possible for that parameter. The highest number was chosen based on if the simulation would not break and if it was large enough so that the simulation doesn’t take long to run. 

## Cell Body Generation
We initialize the starting coordinates for the center of the cell by taking the sum of the body_radius, 〖arm〗_length, and 〖pad〗_radius. We then added a buffer of 0.05 to that sum to make sure that the cell stays within the frame. The following shows the equations used. 

〖max〗_reach= body_radius+〖arm〗_length+〖pad〗_radius+0.05

x_center=〖max〗_reach

y_center=〖max〗_reach

Then finally to add the cell body we utilize the following function. 

scene.add_circle(x_center,y_center,body_radius,-1)

## Cell Arm Generation
The way that we created the arms was we defined the angle between arms to be angle_step. We divided 2π by the number of arms to obtain that. The following shows how we obtain these values. 

angle_step=2π/(num_arms )

Then we decided to discretize the arm by the number of muscles generated by our initialization. By dividing the arm_length by the muscle_count, the discretized arms can be obtained. This variable is thereby known as w_step. The same algorithm is applied for the arm_width. This variable is known as h_step. The following equations show the algorithm. 

w_step=(arm_length)/(muscle_count )

h_step=(arm_width)/(muscle_count )

After we obtain the values of w_step and h_step, then we then run a for loop that essentially creates a muscle along the length of the arm for every arm. The following shows how we do it. 
For every arm, we designate a value for the angle of the arm by doing the following.

angle=i*angle_step
Where i is an index of the arm. 

We figure out the x coordinate and y coordinate by offsetting the x and y center by using trigonometry. The following shows the algorithm that was used in order to perform this. 

arm_x=x_center+cos⁡(angle)*body_radius

arm_y=y_center+sin⁡(angle)*body_radius

Then once we have the coordinates, we can then add the segmented muscles along the arm through the following algorithm. It should be noted that we made each muscle be their own independent actuator. So, the first line that we show is making sure that that happens. This actuator ID is indicated by the value of j. z is the muscle index. 

for z in range (muscle_count):

j=i*muscle_count+z

add_rotated_rect =(arm_x,arm_y,w_step,h_step,j,rotation=angle)

arm_x=arm_x+cos⁡(angle)*(w_step-weld_length)

arm_y=arm_y+sin⁡(angle)*(h_step-weld_length)

So the last two lines above show that we redefine the values for the coordinates of the rectangles arm_x and arm_y for each muscle. Along with that, we can alter this value using weld_length such that the muscles can be imposed on one another. This was done with the intention to make the arms both denser and could actuate in different phases. 

## Cell Pad Generation
At the end of the arm, there should be a pad as well. This was done with the intention such that when the cell lands on the floor, the arm does not rip. So, for each instance of each arm, we must create a pad that appears at the end of the arm. We also must make sure that the semicircle is rotated correctly such that the arc is facing the outside. Therefore for every instance of an arm, we performed the following. 

semi_angle=angle-π/2

scene.add_semicircle(arm_x+cos⁡(angle)*(arm_length)/2,arm_y+sin⁡(angle)*(arm_length)/2,pad_radius,i,rotation=semi_angle)

Now that we were able to create our randomized cells, we were able to observe the different creatures that we created.

## Main portion of the code
In main, we essentially create a population of cells based on our initial parameters, the parameters are made in the following. 

parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=40)  # Number of iterations per simulation
    parser.add_argument('--gens', type=int, default=8)   # Number of generations
    options = parser.parse_args()

In this case we train each species in each generation 40 times. There are also a total of 8 generations in this exmaple.

We run the base diffmpm.py code but we also added in a parameter to do mutations on the fly. The way we did is shown below.

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

## Checkpointing
We also added in a checkpoint system such that if there are errors, you can run it from the last ran generation. It is run through the following

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



## Analyzing Data
At the end of the code, you can see the loss matrices, so we can see how each matrix changes over the course of the iterations. 

What we can do is that we can see how each cell cell performs. It is shown in the following.

            params['final_loss'] = losses[-1]
            params['improvement'] = losses[0] - losses[-1]
            cells_data.append(params)
            all_loss_histories.append(losses)

            # Save loss history to CSV
            loss_csv_path = f"{cell_folder}/loss.csv"
            pd.DataFrame({"Iteration": range(options.iters), "Loss": losses}).to_csv(loss_csv_path, index=False)

        generation_loss_histories.append(all_loss_histories)

Therefore we can view each portion loss history that is associated with each cell.
It should be noted that the more negative the loss, the better that the cell performs in terms of locomotion. 

## Optimization Method

I would initially define the population size for a given generation. Within that generation, each member of the population would be randomly generated and essentially trained based on our fitness model. Our fitness model was slightly modified such that it would measure how close it was to the target both in terms of x and y cartesian coordinates. It would take the sum of these distances as the measure for its loss. 
To change the measurement for its loss I had to integrate the following. 
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


When I implemented this new algorithm, the cool thing about what happened with the behavior of the cells were that they weren’t just trying to roll across the field. Rather they would try to hop to attain both a high y value and a high x value. I did this to differ myself from the diffmpm example as well to observe different behavior than I had seen in the pass. 

Once this was implemented, I would then map out the losses per iteration for each member of the population in each generation. What determined whether they would continue to the next generation was two parameters: Final Performance and Improvement. Final performance was defined as the performance of a given cell in terms of their loss function. While this makes sense as a valid parameter to designate keeping a species into the next generation, there was one piece of information that was missing. If an organism simply performed well enough in the given generation, but didn’t learn anything, then I felt that we would be missing out on important information from the other members of the population. These cells could be designated as those that had a flat, or closely flat line on their loss function. That is why I introduced our second parameter, Improvement. Improvement was essentially defined as how well the cell learned over the iterations. We would take the differences in the losses from the final iteration to the last iteration. Once that was defined, we would then take the top two performers in each of those two categories. If we were to have duplicates that showed up in both, then we would eliminate the duplicate and retain the distinct members. These would then go on to become the parents of the following generation. 
Once the parents of the generation were determined, we implemented a way to perform the mutations. We would have two to four members of the original population go on to become the parents of the following population. To fill out the remaining population size, we would take our survivors, randomly take a parameter that we specified in the table above and alter it randomly. The alteration had to be tuned essentially, because I didn’t want the parameter to be too drastic of a change. Along with that, I didn’t want the change to be too small as it would also cause a drastic bias towards the parent. To properly change the parameters, I implemented the following steps. 

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

There are two instances in which we would want to alter the parameters. The first instance is if we needed to ensure that it was positive integer in the case of num_arms and muscle_count. We essentially added or subtracted a random integer by our chosen parameter. We also still had to ensure that it was within the bounds that we chose. The second instance was to add a value from a normal bell curve that was centered around zero. We chose a value of 0.05 as the standard deviation as that gave us the favorable changes that were not too drastic while still exhibiting a fair difference. This algorithm for implementing a mutation allowed to vary different parameters and see what features helped in creating the locomotion of this creature. The main code will be shown below in the supplementary information. 
