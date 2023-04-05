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


include("../problems/tigerProblem.jl")

# --------------------------simulation-----------------------------
function run_own_sim(package_name, m, policy, n_simulations = 10)
    local rsum = 0
    local b1sum = 0.0
    local nwrong = 0.0
    local ncorrect = 0
    local nlisten = 0
    local nleft = 0
    
    local b1 = Vector{Float64}()
    local b2 = Vector{Float64}()
    local act = Vector{Int64}()

    # run a simulation of our model using the stepthrough function
    local b = initialize_belief(updater(policy), initialstate(m))
    local s = rand(initialstate(m))
    local r_total = 0.0
    local d = 1.0
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
        
        if r != -1
            counter +=1
        end
        #println("state: $s, belief: $([s=>round(pdf(b,s),digits=2) for s in states(m)]), action: $a, obs: $o, reward:$r")
        #println(s, ([s=>round(pdf(b,s),digits=2) for s in states(m)]), o, a, r)

        # store belief and action inex
        push!(b1, round(pdf(b,"left"),digits=2))
        push!(b2, round(pdf(b,"right"),digits=2))
        push!(act, actionindex(m,a))

        # count number of time open the wrong door
        if r ==-100.0
            nwrong += 1
        elseif  r==10.0
            ncorrect += 1
        end

        if a == "listen"
            nlisten += 1
        elseif a =="left"
            nleft += 1
        end

        rsum += r
        r_total += discount(m)*r
        b1sum += round(pdf(b,"left"),digits=2)
        nstep +=1
    end

    

    return n_simulations, b1sum, trunc(Int,rsum), trunc(Int, nlisten),
    trunc(Int, nleft), trunc(Int,nwrong), trunc(Int, ncorrect), trunc(Int, r_total), trunc(Int, nstep)
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
            
            local m = TigerPOMDP()
            local solver = def_solver
            local policy = solve(solver, m)
            # print(policy.alphas)
            
            local n_simulations, b1sum,rsum, nlisten, nleft, nwrong, ncorrect,r_total,n_step = run_own_sim(package_name, m, policy, 1000)
            
            # time taken to execute a given expression or function, in seconds
            local elapsed_time = @elapsed run_own_sim(package_name, m, policy, 1000)
            # -----------------record test result--------------------------
            df = DataFrame(id = i, package = String31(package_name), 
                        n_games = n_simulations, 
                        b1_avg = round(b1sum/n_step,digits=2), 
                        b2_avg= round((n_step-b1sum)/n_step, digits=2), 
                        sum_reward=rsum, 
                        sum_discount_reward = r_total,
                        num_listen=nlisten, num_left = nleft, num_right = n_step-nlisten-nleft,
                        num_correct_door = ncorrect, num_wrong_door = nwrong,
                        avg_steps_per_game = n_step/n_simulations,
                        runtime = elapsed_time)

            append!(old_df,df)
            

        end
    end

    CSV.write(pwd()*"/results/tigerProblem.csv", old_df)
    println("done!")
    #println(old_df)
end


run_solvers()