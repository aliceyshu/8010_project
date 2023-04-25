# using Pkg
# Pkg.add("POMDPs")
# Pkg.add("POMDPModelTools")
# Pkg.add("POMDPSimulators")
# Pkg.add("POMDPPolicies")
# Pkg.add("POMDPLinter")
# Pkg.add("POMDPTools")
# Pkg.add("Parameters")


using POMDPTools: Deterministic, Uniform, SparseCat, Policies, ModelTools
using POMDPs
using POMDPModelTools
using POMDPSimulators
using POMDPPolicies
#using POMDPLinter
using POMDPTools
using Parameters


# -------------------------define Query problem------------------------------
struct QueryPOMDP <: POMDP{String, String, String}
    p_correct::Float64
    indices::Dict{String, Int} 
    QueryPOMDP(p_correct=0.85) = new(p_correct, Dict("11"=>1,"12"=>2,"21"=>3,"13"=>4,"31"=>5,"23"=>6,"32"=>7,"22"=>8,"33"=>9))
end

# 9 states in total (loaded, loaded),(loaded, unloaded)...(loaded,down)...(down,down)
# 1=loaded, 2=unloaded, 3=down
POMDPs.states(m::QueryPOMDP) = ["11","12","21","13","31","23","32","22","33"]
# query 1, query 2
POMDPs.actions(m::QueryPOMDP) = ["1","2"]
# no response, fast response, slow response
POMDPs.observations(m::QueryPOMDP) = ["1","2","3"]

POMDPs.discount(m::QueryPOMDP) = 0.95
POMDPs.stateindex(m::QueryPOMDP, ss::String) = m.indices[ss]
POMDPs.actionindex(m::QueryPOMDP, aa) = parse(Int64, aa)
POMDPs.obsindex(m::QueryPOMDP, oo) = parse(Int64, oo)


# Define the transition function
function POMDPs.transition(m::QueryPOMDP, state, action)
    # Check the current state and action to determine the probability of transitioning to a new state
    p2 = (1-m.p_correct)/2
    action = parse(Int64, action)
    if state=="11"
        if action == 1
            return SparseCat(["11","21","31"],[m.p_correct, p2, p2])
        elseif action == 2
            return SparseCat(["11","12","13"],[m.p_correct, p2, p2])
        end
    elseif state =="12"
        if action == 1
            return SparseCat(["12","22","32"],[m.p_correct, p2, p2])
        elseif action == 2
            return SparseCat(["12","22","12"],[p2,m.p_correct, p2])
        end
    elseif state=="13"
        if action == 1
            return SparseCat(["13","23","33"],[p2,m.p_correct, p2])
        elseif action == 2
            return SparseCat(["33","23","13"],[m.p_correct, p2, p2])
        end
    elseif state=="21"
        if action == 1
            return SparseCat(["12","21","31"],[p2,m.p_correct, p2])
        elseif action == 2
            return SparseCat(["31","21","11"],[m.p_correct, p2, p2])
        end
    elseif state =="22"
        if action == 1
            return SparseCat(["22","12","32"],[m.p_correct, p2, p2])
        elseif action == 2
            return SparseCat(["22","21","23"],[m.p_correct, p2, p2])
        end
    elseif state =="23"
        if action == 1
            return SparseCat(["23","33","13"],[m.p_correct, p2, p2])
        elseif action == 2
            return SparseCat(["33","32","31"],[m.p_correct, p2, p2])
        end
    elseif state=="31"
        if action == 1
            return SparseCat(["13","31","21"],[p2,m.p_correct, p2])
        elseif action == 2
            return SparseCat(["31","32","33"],[m.p_correct, p2, p2])
        end
    elseif state=="32"
        if action == 1
            return SparseCat(["23","32","12"],[p2,m.p_correct, p2])
        elseif action == 2
            return SparseCat(["32","33","31"],[m.p_correct, p2, p2])
        end
    elseif state=="33"
        # Both servers are down, so they cannot respond to any queries
        return SparseCat(["33"],[1.0])
    end
end


# Define the observation function
function POMDPs.observation(m::QueryPOMDP, action, next_state)
    p2 = (1-m.p_correct)/2
    # Check the current state, action, and next state to determine the probability of observing each observation
    if next_state=="33"
        # Both servers are down, so there can be no response
        return SparseCat(["1"],[1.0])
    elseif next_state=="11"
        # Both servers are loaded, so there is a high probability of getting a fast response and a low probability of getting a slow response or no response
        return SparseCat(["2","3","1"],[m.p_correct, p2, p2])
        
    elseif next_state =="22"
        # Both servers are unloaded, so there is a low probability of getting a fast response and a high probability of getting a slow response or no response
        return SparseCat(["2","3","1"],[p2, m.p_correct, p2])

    else
        # One server is loaded and the other is unloaded or down, so there is a moderate probability of getting a fast response and a moderate probability of getting a slow response or no response
        return SparseCat(["2","3","1"],[0.4, 0.4, 0.2])
    end
end


function POMDPs.reward(m::QueryPOMDP, s, a, o)
    if o == "1" # no response
        return 0.0
    elseif o == "3" # query was slow
        return 3.0
    else # query was fast
        return 10.0
    end
end


function POMDPs.initialstate(m::QueryPOMDP) 
    s = states(m)
    ns = length(s)
    p = zeros(ns) .+ 1.0 / (ns)
    return SparseCat(s,p)
end

