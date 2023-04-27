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
#Pkg.add(PackageSpec(name="IncrementalPruning", version="0.2.0"))
=#


using DataFrames
using CSV
using ElectronDisplay
using POMDPs
using POMDPModels
using ParticleFilters
using POMDPLinter:@POMDP_require
using POMDPTools
using POMDPModelTools

using QMDP
using FIB
using PointBasedValueIteration
using SARSOP
using IncrementalPruning
using BasicPOMCP
using POMCPOW
using Random

# include("../problems/tMazeProblem.jl")

# POMDPs.isterminal(m::POMDP, s) = (s==terminalstate)
function POMDPs.reward(m::TMaze, s, a, sp) 
    if isterminal(m,s) != true 
        r= reward(m, s, a)
    else
        r = -100 #placeholder
    end

    return r
end


POMDPs.isterminal(mdp::UnderlyingMDP{P, S, A}, s::TerminalState) where {P, S, A} = true
POMDPs.isterminal(mdp::UnderlyingMDP{P, S, A}, s::TMazeState) where {P, S, A} = false
POMDPs.isterminal(m::Union{TMaze, MDP, POMDP}, s::TMazeState) = false
POMDPs.isterminal(m::Union{TMaze, MDP,POMDP}, s::TerminalState) = true


# --------------------------simulation-----------------------------
function run_tMaze_sim(package_name, m, policy, n_simulations = 10,p=false)
    
    function u(policy,m)
        if package_name == "POMCPOW" ||package_name == "POMCP"
            return BootstrapFilter(m,1000)
        else
            return updater(policy)
        end
    end
    
    # run a simulation of our model using the stepthrough function
    local rsum = 0
    local s = rand(initialstate(m))
    local b = initialize_belief(u(policy,m), initialstate(m))

    local d = 1.0
    local r_total = 0.0
    local counter = 1.0
    local nstep = 0.0
    local n=0

    #  while !isterminal(m, s)
    while counter <= n_simulations

        a = action(policy, b)
        s, o, r = @gen(:sp,:o,:r)(m, s, a)
        # println(a, s.vals[2])
        #r = reward(m, s,a)
        #o = observation(m, s)

        rsum += r
        d *= discount(m)
        r_total += d*r
        b = update(u(policy,m), b, a, o)


        if p==true
            # println(s.x)
            if s!=TerminalState()
                println("state: $(s.x), action: $a, obs: $o,val, reward:$r")
            else
                println("state: $s, action: $a, obs: $o,val, reward:$r")
            end
            #println(s, ([s=>round(pdf(b,s),digits=2) for s in states(m)]), o, a, r)
        end

        # we want to stop iteration after 1000 steps, some solver does not work well with tMaze tMazeSample
        # usually they will keep move forward and and have total reward of 0 (though 
        # they might find a tMaze by chance, this happens rarely)
        # to avoid this, we set this limitation of stop current game after 1000 steps
        # println(isterminal(m,s))
        #if s == TerminalState() || mod(nstep,1000)==0
        if isterminal(m,s) == true || n ==100
            counter +=1
            # println(".")
            n=0
            s = rand(initialstate(m))
            b = initialize_belief(u(policy,m), initialstate(m))
            
        end
        #counter+=1
        
        n+=1
        nstep +=1
        #print(nstep)
    end

    

    return n_simulations, trunc(Int,rsum),  trunc(Int, r_total), trunc(Int, nstep)
end

function run_tMaze_solvers(p=false,n_sim=1000,n_round=10)
    # ---------------------------solvers--------------------------------
    old_df =  DataFrame()

    solver_dict = Dict(
        "POMCP" => POMCPSolver(tree_queries=100, default_action = 2), # ok
        "POMCPOW" => POMCPOWSolver(tree_queries=100, default_action = 2), #ok
        "QMDP" => QMDPSolver(max_iterations=20,belres=1e-3),
        "FIB" => FIBSolver(),#ok
        "PBVI" => PBVISolver(), #ok
        "SARSOP" => SARSOPSolver(precision=1e-3, verbose = false),
        #"IP" =>  PruneSolver(),
        
        )
    

    println("begin...")
    for (package_name, def_solver) in solver_dict
        println(package_name)
        for i in 1:n_round
            println("round $i")
        
            local m= TMaze(n=10)

            local solver = def_solver
            local policy = solve(solver, m)
            # print(policy.alphas)
            
            local n_simulations, rsum, r_total,n_step = run_tMaze_sim(package_name, m, policy, n_sim, p)
            
            # time taken to execute a given expression or function, in seconds
            local elapsed_time = @elapsed run_tMaze_sim(package_name, m, policy, n_sim,p)
            # -----------------record test result--------------------------
            df = DataFrame(id = i, package = String31(package_name), 
                        n_games = n_simulations, 
                        sum_reward=rsum, 
                        sum_discount_reward = r_total,
                        avg_steps_per_game = n_step/n_simulations,
                        runtime = elapsed_time)
            # println(elapsed_time)
            append!(old_df,df)
            # println(rsum)

        end
    end
    
    if p==false && n_sim ==1000
        CSV.write(pwd()*"/results/tMazeProblem.csv", old_df)
    end
    
    println("done!")
    #println(old_df)
end

# print or not, how many games, repeat for how many times
#run_tMaze_solvers(false, 10,1)
run_tMaze_solvers(false, 1000,10)