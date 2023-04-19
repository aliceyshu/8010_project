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
using POMDPModels

using QMDP
using FIB
using PointBasedValueIteration
using SARSOP
#using IncrementalPruning
using BasicPOMCP
using POMCPOW
using Random




# --------------------------simulation-----------------------------
function run_hallway_sim(package_name, m, policy, n_simulations = 10)

    # run a simulation of our model using the stepthrough function
    local b = initialize_belief(updater(policy), initialstate(m))
    local s = rand(initialstate(m))
    local r_total = 0.0
    local d = 1.0
    local counter = 1.0
    local nstep = 0.0
    local rsum = 0.0

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
        
        if r != -1
            counter +=1
        end
        println("state: $s, belief: $([s=>round(pdf(b,s),digits=2) for s in states(m)]), action: $a, obs: $o, reward:$r")
        #println(s, ([s=>round(pdf(b,s),digits=2) for s in states(m)]), o, a, r)

        

        rsum += r
        r_total += discount(m)*r
        nstep +=1
    end

    

    return n_simulations, trunc(Int,rsum), trunc(Int, r_total), trunc(Int, nstep)
end

function run_hallway_solvers()
    # ---------------------------solvers--------------------------------
    old_df =  DataFrame()


    solver_dict = Dict(
        "POMCP" => POMCPSolver(tree_queries=100, rng=MersenneTwister(123),default_action=1),
        #"POMCPOW" => POMCPOWSolver(default_action=1),
        "QMDP" => QMDPSolver(),
        "FIB" => FIBSolver(),
        #"PBVI" => PBVISolver(),
        "SARSOP" => SARSOPSolver(precision=1e-3, verbose = false),
        #"IP" => PruneSolver(),
        
        )


    println("begin...")
    for (package_name, def_solver) in solver_dict
        println(package_name)
        for i in 1:10
            println("round $i")
            
            local m = MiniHallway()
            local n_states = length(states(m))
            local solver = def_solver
            local policy = solve(solver, m)
            # print(policy.alphas)
            
            local n_simulations,rsum, r_total,n_step = run_hallway_sim(package_name, m, policy, 1000)
            
            # time taken to execute a given expression or function, in seconds
            local elapsed_time = @elapsed run_hallway_sim(package_name, m, policy, 1000)
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

    CSV.write(pwd()*"/results/miniHallwayProblem.csv", old_df)
    println("done!")
    #println(old_df)
end


run_hallway_solvers()