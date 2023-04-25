using RockSample
using POMDPGifs
using Cairo
using POMDPs
using FIB

m=RockSamplePOMDP(map_size = (3,3),
                rocks_positions=[(2,2),(1,2)], 
                init_pos = (1,1),
                sensor_efficiency=20.0,
                discount_factor=0.95, 
                good_rock_reward = 10.0,
                sensor_use_penalty=-1.0,
                exit_reward= 20.0)

solver = FIBSolver()
policy = solve(solver, m)
sim = GifSimulator(filename="problems_simulation/rockSampleProblem_viz.png", max_steps=1)
simulate(sim, m, policy)