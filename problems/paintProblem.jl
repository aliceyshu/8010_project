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


# -------------------------define tiger problem------------------------------
struct PaintPOMDP <: POMDP{String, String, String}
    p_correct::Float64
    indices::Dict{String, Int}

    PaintPOMDP(p_correct=0.85) = new(p_correct, Dict("left"=>1, "right"=>2, "listen"=>3))
end


POMDPs.states(m::PaintPOMDP) = ["left", "right"]
POMDPs.actions(m::PaintPOMDP) = ["left", "right", "listen"]
POMDPs.observations(m::PaintPOMDP) = ["left", "right"]
POMDPs.discount(m::PaintPOMDP) = 0.95
POMDPs.stateindex(m::PaintPOMDP, s) = m.indices[s]
POMDPs.actionindex(m::PaintPOMDP, a) = m.indices[a]
POMDPs.obsindex(m::PaintPOMDP, o) = m.indices[o]


function POMDPs.transition(m::PaintPOMDP, s, a)
    if a == "listen"
        return Deterministic(s)           # tiger stays behind the same door
    else                                  # a door is opened
        return Uniform(["left", "right"]) # reset
    end
end


function POMDPs.observation(m::PaintPOMDP, a, sp)
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


function POMDPs.reward(m::PaintPOMDP, s, a)
    if a == "listen"
        return -1.0
    elseif s == a # the tiger was found
        return -100.0
    else # the tiger was escaped
        return 10.0
    end
end


POMDPs.initialstate(m::PaintPOMDP) = Uniform(["left", "right"])

