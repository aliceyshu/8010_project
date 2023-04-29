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


# -------------------------define Paint problem------------------------------
struct PaintPOMDP <: POMDP{String, String, String}
    p_correct::Float64
    indices::Dict{String, Int} 
    PaintPOMDP(p_correct=0.85) = new(p_correct, Dict("NFL-NBL-NPA"=>1,"NFL-NBL-PA"=>2,"FL-NBL-PA"=>3,"FL-BL-NPA"=>4))
end

# flawed or not - blemished or not - painted or not
POMDPs.states(m::PaintPOMDP) = ["NFL-NBL-NPA","NFL-NBL-PA","FL-NBL-PA","FL-BL-NPA"]
# Paint, inspect, ship, reject
POMDPs.actions(m::PaintPOMDP) = ["1","2","3","4"]
# not fully observed, inspect action can help reduce uncertainty
# blemished or not, 1=non-blemished, 2=blemished
POMDPs.observations(m::PaintPOMDP) = ["1","2"]

POMDPs.discount(m::PaintPOMDP) = 0.95
POMDPs.stateindex(m::PaintPOMDP, ss::String) = m.indices[ss]
POMDPs.actionindex(m::PaintPOMDP, aa) = parse(Int64, aa)
POMDPs.obsindex(m::PaintPOMDP, oo) = parse(Int64, oo)


# Define the transition function
function POMDPs.transition(m::PaintPOMDP, s, a)
    p2= (1-m.p_correct)/3
    # Paint
    if a == "1"
        if s == states(m)[1]
            return SparseCat(["NFL-NBL-NPA","NFL-NBL-PA","FL-NBL-PA","FL-BL-NPA"],[1-m.p_correct,m.p_correct,0.0,0.0])
        elseif s==states(m)[2]
            return SparseCat(["NFL-NBL-PA"],[1.0])
        elseif s==states(m)[3]
            return SparseCat(["FL-NBL-PA"],[1.0])
        else
            return SparseCat(["NFL-NBL-NPA","NFL-NBL-PA","FL-NBL-PA","FL-BL-NPA"],[0.0, 0.0, m.p_correct,1-m.p_correct])
        end
    # inspect
    elseif a != "2"
        if s == states(m)[1]
            return SparseCat(states(m),[m.p_correct,1-m.p_correct,0,0])
        elseif s == states(m)[2]
            return SparseCat(states(m),[p2,m.p_correct,p2,p2])
        elseif s == states(m)[3]
            return SparseCat(states(m),[p2,p2,m.p_correct,p2])
        else 
            return SparseCat(states(m),[p2,p2,p2,m.p_correct])
        end
    # ship & reject
    else
        # restart
        return SparseCat(states(m),[0.5,0.0,0.0,0.5])
    end
end


# Define the observation function
function POMDPs.observation(m::PaintPOMDP, action, next_state)
    # 1=non-blemished, 2=blemished
    # paint
    if action == "1"
        return SparseCat(["1"],[1.0])
    # inspect
    elseif action == "2"
        if next_state ==states(m)[1] || next_state ==states(m)[2] || next_state == states(m)[3]
            return SparseCat(["1","2"],[0.75,0.25])
        else
            return SparseCat(["1","2"],[0.25,0.75])
        end
    # ship
    elseif action == "3"
        return SparseCat(["1"],[1.0])
    # reject
    else
        return SparseCat(["1"],[1.0])
    end
end


function POMDPs.reward(m::PaintPOMDP, s, a)
    if (s == states(m)[2]) && (a == actions(m)[3])
        reward = 1.0
    elseif (s == states(m)[4]) && (a == actions(m)[4])
        reward = 1.0
    else
        reward = -1.0
    end

    return reward
end


function POMDPs.initialstate(m::PaintPOMDP) 
    s = states(m)
    p = [0.5, 0.0, 0.0, 0.5]
    return SparseCat(s,p)
end

