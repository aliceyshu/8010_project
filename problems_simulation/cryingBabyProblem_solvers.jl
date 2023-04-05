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



# --------------------------simulation-----------------------------
function run_own_sim(package_name, m, policy, n_simulations = 10)
    local rsum = 0
    
    local b1 = Vector{Float64}()
    local b2 = Vector{Float64}()
    local act = Vector{Int64}()

    # run a simulation of our model using the stepthrough function
    local b = initialize_belief(updater(policy), initialstate(m))
    local s = rand(initialstate(m))
    local d = 1.0

    local r_total = 0.0
    local counter = 1.0
    local nstep = 0.0

    local nfeed = 0.0
    local nhungry = 0.0
    local ncrying= 0.0

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
        
        if r==-5.0 || r===15.0
            counter +=1
        end
        # println("state: $s, belief: $([s=>round(pdf(b,s),digits=2) for s in states(m)]), action: $a, obs: $o, reward:$r")
        #println(s, ([s=>round(pdf(b,s),digits=2) for s in states(m)]), o, a, r)

        # store belief and action inex
        #push!(b1, round(pdf(b,"left"),digits=2))
        #push!(b2, round(pdf(b,"right"),digits=2))
        #push!(act, actionindex(m,a))

        # count number of time open the wrong door
        
        if r ==-15.0
            nfeed += 1
            nhungry +=1
        elseif  r==-5.0
            nfeed += 1
        elseif  r==-10.0
            nhungry += 1
        end

        if o == true
            ncrying += 1
        end
        

        rsum += r
        r_total += discount(m)*r

        nstep +=1
    end

    

    return n_simulations, trunc(Int,rsum),  trunc(Int, r_total), trunc(Int, nstep), trunc(Int, nfeed),trunc(Int, nhungry), trunc(Int, ncrying)
end

function run_solvers()
    # ---------------------------solvers--------------------------------
    old_df =  DataFrame()


    solver_dict = Dict(
        "QMDP" => QMDPSolver(),
        "FIB" => FIBSolver(),
        "PBVI" => PBVISolver(),
        "SARSOP" => SARSOPSolver(verbose=false)
        
        )

    #=
    solver_dict =  Dict(
        "monahan_own_solver" => monahanSolver(max_iterations=10)
        )
    =#

    println("begin...")
    for (package_name, def_solver) in solver_dict
        println(package_name)
        for i in 1:100
            println("round $i")
            
            local m = BabyPOMDP()
            local solver = def_solver
            local policy = solve(solver, m)
            # print(policy.alphas)
            
            local n_simulations, rsum, r_total,n_step, nfeed, nhungry, ncrying = run_own_sim(package_name, m, policy, 1000)
            
            # time taken to execute a given expression or function, in seconds
            local elapsed_time = @elapsed run_own_sim(package_name, m, policy, 1000)
            # -----------------record test result--------------------------
            df = DataFrame(id = i, package = String31(package_name), 
                        n_games = n_simulations, 
                        sum_reward=rsum, 
                        sum_discount_reward = r_total,
                        avg_steps_per_game = n_step/n_simulations,
                        runtime = elapsed_time,
                        n_feed = nfeed, n_hungry=nhungry, n_crying = ncrying)

            append!(old_df,df)
            

        end
    end

    CSV.write(pwd()*"/results/cryingBabyProblem.csv", old_df)
    println("done!")
    #println(old_df)
end


run_solvers()