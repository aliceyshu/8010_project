#=
using Pkg
Pkg.add("POMDPTools")
Pkg.add("POMDPs")
Pkg.add("POMDPModelTools")
Pkg.add("POMDPSimulators")
Pkg.add("POMDPPolicies")
=#


using POMDPTools: Deterministic, Uniform, SparseCat, Policies
using POMDPs
using POMDPModelTools
using POMDPSimulators
using POMDPPolicies
using POMDPTools


# -------------------------define Query problem------------------------------
struct QueryPOMDP <: POMDP{String, String, String}
    p_correct::Float64
    indices::Dict{String, Int}

    QueryPOMDP(p_correct=0.85) = new(p_correct, Dict("left"=>1, "right"=>2, "listen"=>3))
end

# 9 states in total (loaded, loaded),(loaded, unloaded)...(loaded,down)...(down,down)
# 1=loaded, 2=unloaded, 3=down
POMDPs.states(m::QueryPOMDP) = [(1,1),(1,2),(2,1),(1,3),(3,1),(2,3),(3,2)(2,2),(3,3)]
# query 1, query 2
POMDPs.actions(m::QueryPOMDP) = 1:2
# no response, fast response, slow response
POMDPs.observations(m::QueryPOMDP) = 1:3

POMDPs.discount(m::QueryPOMDP) = 0.95
POMDPs.stateindex(m::QueryPOMDP, s) = m.indices[s]
POMDPs.actionindex(m::QueryPOMDP, a) = m.indices[a]
POMDPs.obsindex(m::QueryPOMDP, o) = m.indices[o]





# Define the transition function
function transition(m::QueryPOMDP, state, action)
    # Check the current state and action to determine the probability of transitioning to a new state
    if state==(1,1)
        if action == 1
            return SparseCat((1,1)=>0.9, (2,1)=>0.05, (3,1)=>0.05)
        elseif action == 2
            return SparseCat((1,1)=>0.9, (1,2)=>0.05, (1,3)=>0.05)
        end
    elseif state ==(1,2)
        if action == 1
            return SparseCat((1,2)=>0.9, (2,2)=>0.05, (3,2)=>0.05)
        elseif action == 2
            return SparseCat((1,2)=>0.05, (2,2)=>0.9, (1,2)=>0.05)
        end
    elseif state==(1,3)
        if action == 1
            return SparseCat((1,3)=>0.05,(2,3)=>0.9, (3,3)=>0.05)
        elseif action == 2
            return SparseCat((3,3)=>0.9,(2,3)=>0.05, (1,3)=>0.05)
        end
    elseif state==(2,1)
        if action == 1
            return SparseCat((1,2)=>0.05, (2,1)=>0.9, (3,1)=>0.05)
        elseif action == 2
            return SparseCat((3,1)=>0.9, (2,1)=>0.05, (1,1)=>0.05)
        end
    elseif state ==(2,2)
        if action == 1
            return SparseCat((2,2)=>0.9, (1,2)=>0.05, (3,2)=>0.05)
        elseif action == 2
            return SparseCat((2,2)=>0.9, (2,1)=>0.05,(2,3)=>0.05)
        end
    elseif state ==(2,3)
        if action == 1
            return SparseCat((2,3)=>0.9, (3,3)=>0.05, (1,3)=>0.05)
        elseif action == 2
            return SparseCat((3,3)=>0.9, (3,2)=>0.05, (3,1)=>0.05)
        end
    elseif state==(3,1)
        if action == 1
            return SparseCat((1,3)=>0.05, (3,1)=>0.9, (2,1)=>0.05)
        elseif action == 2
            return SparseCat((3,1)=>0.9, (3,2)=>0.05, (3,3)=>0.05)
        end
    elseif state==(3,2)
        if action == 1
            return SparseCat((2,3)=>0.05, (3,2)=>0.9, (1,2)=>0.05)
        elseif action == 2
            return SparseCat((3,2)=>0.9, (3,3)=>0.05, (3,1)=>0.05)
        end
    elseif state==(3,3)
        # Both servers are down, so they cannot respond to any queries
        return SparseCat((3,3)=>1.0)
    end
end

function POMDPs.observation(m::QueryPOMDP, a, sp)
    if a == "listen"
        if sp == "left"
            return SparseCat(["left", "right"], [m.p_correct, 1.0-m.p_correct])
        else
            return SparseCat(["right", "left"], [m.p_correct, 1.0-m.p_correct])
        end
    else
        return Uniform(["left", "right"])
    end
end

# Define the observation function
function observation(m::QueryPOMDP, action, next_state)
    # Check the current state, action, and next state to determine the probability of observing each observation
    if next_state[1] == 3 && next_state[2] == 3
        # Both servers are down, so there can be no response
        return SparseCat(1 => 1.0)
    elseif next_state[1] == 1 && next_state[2] == 1
        # Both servers are loaded, so there is a high probability of getting a fast response and a low probability of getting a slow response or no response
        return SparseCat(2 => 0.9, 3 => 0.05, 1 => 0.05)
        
    elseif next_state[1] == 2 && next_state[2] == 2
        # Both servers are unloaded, so there is a low probability of getting a fast response and a high probability of getting a slow response or no response
        return SparseCat(2 => 0.05, 3 => 0.9, 1 => 0.05)

    else
        # One server is loaded and the other is unloaded or down, so there is a moderate probability of getting a fast response and a moderate probability of getting a slow response or no response
        return SparseCat(2 => 0.4, 3 => 0.4, 1 => 0.2)
    end
end


function POMDPs.reward(m::QueryPOMDP, s, a,o)
    if o == 1 # no response
        return 0.0
    elseif o == 2 # query was slow
        return 3.0
    else # query was fast
        return 10.0
    end
end


function initialstate(m::QueryPOMDP) 
    s = states(m)
    ns = length(s)
    p = zeros(ns) .+ 1.0 / (ns)
    return SparseCat(s,p)
end


