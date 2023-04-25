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

using JLD2
using DataFrames
using CSV
using ElectronDisplay
using POMDPs
#using POMDPModels
using ParticleFilters
using POMDPLinter:@POMDP_require

using QMDP
using FIB
using PointBasedValueIteration
using SARSOP
using IncrementalPruning
using BasicPOMCP
using POMCPOW
using Random


# --------------------------simulation-----------------------------
function run_query_sim(package_name, m, policy, n_simulations = 10,p=false)
    
    function u(policy,m)
        if package_name == "POMCP" || package_name == "POMCPOW"
            return BootstrapFilter(m,10)
        else
            return updater(policy)
        end
    end

    # run a simulation of our model using the stepthrough function
    local rsum = 0
    local b = initialize_belief(u(policy,m), initialstate(m))
    local s = rand(initialstate(m))
    local d = 1.0

    local r_total = 0.0
    local counter = 0.0
    local nstep = 0.0

    #  while !isterminal(m, s)
    while counter <= n_simulations
        if mod(counter,10) == 0
            #println(counter)
        end

        a = action(policy, b)
        s, o, r = @gen(:sp,:o,:r)(m, s, a)

        rsum += r
        d *= discount(m)
        r_total += d*r
        b = update(u(policy,m), b, a, o)
        


        if p==true
            println("state: $s, action: $a, obs: $o, reward:$r")
        #println(s, ([s=>round(pdf(b,s),digits=2) for s in states(m)]), o, a, r)
        end


        # we want to stop iteration after 1000 steps, some solver does not work well with query querySample
        # usually they will keep move forward and and have total reward of 0 (though 
        # they might find a query by chance, this happens rarely)
        # to avoid this, we set this limitation of stop current game after 1000 steps
        if (r == 20) || mod(nstep,1000)==0
            counter +=1
            #println(".")
            b = initialize_belief(updater(policy), initialstate(m))
            s = rand(initialstate(m))
        end
        #counter+=1


        nstep +=1
        #print(nstep)
    end

    

    return n_simulations, trunc(Int,rsum),  trunc(Int, r_total), trunc(Int, nstep)
end

function run_query_solvers(p=false,n_sim=1000,n_round=10)
    # ---------------------------solvers--------------------------------
    old_df =  DataFrame()


    
    solver_dict = Dict(
        "POMCP" => POMCPSolver(tree_queries=100, default_action = 1),
        "POMCPOW" => POMCPOWSolver(tree_queries=100, default_action = 1),
        "QMDP" => QMDPSolver(max_iterations=20,belres=1e-3),
        "FIB" => FIBSolver(),
        "PBVI" => PBVISolver(),
        "SARSOP" => SARSOPSolver(precision=1e-3, verbose = false),
        #"IP" =>  PruneSolver(),
        
        )

    println("begin...")
    for (package_name, def_solver) in solver_dict
        println(package_name)
        for i in 1:n_round
            println("round $i")
            

            local m=QueryPOMDP()
                
            local solver = def_solver
            local policy = solve(solver, m)
            # print(policy.alphas)
            
            local n_simulations, rsum, r_total,n_step = run_query_sim(package_name, m, policy, n_sim, p)
            
            # time taken to execute a given expression or function, in seconds
            local elapsed_time = @elapsed run_query_sim(package_name, m, policy, n_sim,p)
            # -----------------record test result--------------------------
            df = DataFrame(id = i, package = String31(package_name), 
                        n_games = n_simulations, 
                        sum_reward=rsum, 
                        sum_discount_reward = r_total,
                        avg_steps_per_game = n_step/n_simulations,
                        runtime = elapsed_time)

            append!(old_df,df)
            # println(rsum)

        end
    end

    CSV.write(pwd()*"/results/querySampleProblem.csv", old_df)
    println("done!")
    #println(old_df)
end

# print or not, how many games, repeat for how many times
run_query_solvers(true, 10,1)
# run_query_solvers(false, 1000,10)