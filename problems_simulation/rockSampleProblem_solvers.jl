#=
using Pkg

Pkg.add("JLD2")
Pkg.add("DataFrames")
Pkg.add("CSV")
Pkg.add("ElectronDisplay")
Pkg.add("POMDPs")

Pkg.add("QMDP")
Pkg.add("FIB")
Pkg.add("PointBasedValueIteration")
Pkg.add("SARSOP")
=#

using JLD2
using DataFrames
using CSV
using ElectronDisplay
using POMDPs
#using POMDPModels

using QMDP
using FIB
using PointBasedValueIteration
using SARSOP
using IncrementalPruning
using BasicPOMCP
using POMCPOW
using Random


include("../problems/rockSampleProblem.jl")

# --------------------------simulation-----------------------------
function run_rock_sim(package_name, m, policy, n_simulations = 10)
    local rsum = 0
    


    # run a simulation of our model using the stepthrough function
    local b = initialize_belief(updater(policy), initialstate(m))
    local s = rand(initialstate(m))
    local d = 1.0

    local r_total = 0.0
    local counter = 1.0
    local nstep = 0.0

    #  while !isterminal(m, s)
    while counter <= n_simulations
        if mod(counter,10) == 0
            #println(counter)
        end

        a = action(policy, b)
        s, o, r = @gen(:sp,:o,:r)(m, s, a)

        r_total += d*r
        d *= discount(m)
        b = update(updater(policy), b, a, o)
        
        if r == 15
            counter +=1
            println(".",s)
            b = initialize_belief(updater(policy), initialstate(m))
            s = rand(initialstate(m))
        end
        #println("state: $s, belief: $([s=>round(pdf(b,s),digits=2) for s in states(m)]), action: $a, obs: $o, reward:$r")
        #println(s, ([s=>round(pdf(b,s),digits=2) for s in states(m)]), o, a, r)

        

        rsum += r
        r_total += discount(m)*r

        nstep +=1
    end

    

    return n_simulations, trunc(Int,rsum),  trunc(Int, r_total), trunc(Int, nstep)
end

function run_rock_solvers()
    # ---------------------------solvers--------------------------------
    old_df =  DataFrame()


    solver_dict = Dict(
        "POMCP" => POMCPSolver(tree_queries=100, rng=MersenneTwister(123), default_action = 1),
        "POMCPOW" => POMCPOWSolver(tree_queries=100, default_action = 1),
        #"QMDP" => QMDPSolver(),
        #"FIB" => FIBSolver(),
        #"PBVI" => PBVISolver(),
        "SARSOP" => SARSOPSolver(precision=1e-3, verbose = false),
        #"IP" =>  PruneSolver(),
        
        )
    #=
        
    =#

    println("begin...")
    for (package_name, def_solver) in solver_dict
        println(package_name)
        for i in 1:10
            println("round $i")
            
            local m = RockPOMDP{2}(
                map_size = (3,3),
                rocks_positions=[(2,2),(1,2)], 
                init_pos = (1,1),
                sensor_efficiency=20.0,
                discount_factor=0.95, 
                good_rock_reward = 20.0)
            local solver = def_solver
            local policy = solve(solver, m)
            # print(policy.alphas)
            
            local n_simulations, rsum, r_total,n_step = run_rock_sim(package_name, m, policy, 1000)
            
            # time taken to execute a given expression or function, in seconds
            local elapsed_time = @elapsed run_rock_sim(package_name, m, policy, 1000)
            # -----------------record test result--------------------------
            df = DataFrame(id = i, package = String31(package_name), 
                        n_games = n_simulations, 
                        sum_reward=rsum, 
                        sum_discount_reward = r_total,
                        avg_steps_per_game = n_step/n_simulations,
                        runtime = elapsed_time)

            append!(old_df,df)
            

        end
    end

    CSV.write(pwd()*"/results/rockSampleProblem.csv", old_df)
    println("done!")
    #println(old_df)
end


run_rock_solvers()