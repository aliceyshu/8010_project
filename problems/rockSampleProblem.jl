#=
using Pkg
Pkg.add("LinearAlgebra")
Pkg.add("POMDPs")
Pkg.add("POMDPTools")
Pkg.add("StaticArrays")
Pkg.add("Parameters")
Pkg.add("Random")
Pkg.add("Compose")
Pkg.add("Combinatorics")
Pkg.add("DiscreteValueIteration")
Pkg.add("ParticleFilters")
=#

using LinearAlgebra
using POMDPs
using POMDPTools
using StaticArrays
using Parameters
using Random
using Compose
using Combinatorics
using DiscreteValueIteration
using ParticleFilters           # used in heuristics
using POMDPModelChecking

const RSPos = SVector{2, Int}

"""
    RSState{K}
Represents the state in a RockPOMDP problem. 
`K` is an integer representing the number of rocks
# Fields
- `pos::RPos` position of the robot
- `rocks::SVector{K, Bool}` the status of the rocks (false=bad, true=good)
"""
struct RSState{K}
    pos::RSPos 
    rocks::SVector{K, Bool}
end


@with_kw struct RockPOMDP{K} <: POMDP{RSState{K}, Int, Int}
    map_size::Tuple{Int, Int} = (5,5)
    rocks_positions::SVector{K,RSPos} = @SVector([(1,1), (3,3), (4,4)])
    init_pos::RSPos = (1,1)
    sensor_efficiency::Float64 = 20.0
    bad_rock_penalty::Float64 = -10
    good_rock_reward::Float64 = 10.
    step_penalty::Float64 = 0.
    sensor_use_penalty::Float64 = 0.
    exit_reward::Float64 = 15.
    terminal_state::RSState{K} = RSState(RSPos(-1,-1),
                                         SVector{length(rocks_positions),Bool}(falses(length(rocks_positions))))
    # Some special indices for quickly retrieving the stateindex of any state
    indices::Vector{Int} = cumprod([map_size[1], map_size[2], fill(2, length(rocks_positions))...][1:end-1])
    discount_factor::Float64 = 0.95
end

# to handle the case where rocks_positions is not a StaticArray
function RockPOMDP(map_size,
                         rocks_positions,
                         args...
                        )

    k = length(rocks_positions)
    return RockPOMDP{k}(map_size,
                              SVector{k,RSPos}(rocks_positions),
                              args...
                             )
end

# Generate a random instance of RockSample(n,m) with a n×n square map and m rocks
RockPOMDP(map_size::Int, rocknum::Int, rng::AbstractRNG=Random.GLOBAL_RNG) = RockPOMDP((map_size,map_size), rocknum, rng)

# Generate a random instance of RockSample with a n×m map and l rocks
function RockPOMDP(map_size::Tuple{Int, Int}, rocknum::Int, rng::AbstractRNG=Random.GLOBAL_RNG)
    possible_ps = [(i, j) for i in 1:map_size[1], j in 1:map_size[2]]
    selected = unique(rand(rng, possible_ps, rocknum))
    while length(selected) != rocknum
        push!(selected, rand(rng, possible_ps))
        selected = unique!(selected)
    end
    return RockPOMDP(map_size=map_size, rocks_positions=selected)
end

# transform a Rocksample state to a vector 
function POMDPs.convert_s(T::Type{<:AbstractArray}, s::RSState, m::RockPOMDP)
    return convert(T, vcat(s.pos, s.rocks))
end

# transform a vector to a RSState
function POMDPs.convert_s(T::Type{RSState}, v::AbstractArray, m::RockPOMDP)
    return RSState(RSPos(v[1], v[2]), SVector{length(v)-2,Bool}(v[i] for i = 3:length(v)))
end


# To handle the case where the `rocks_positions` is specified
RockPOMDP(map_size::Tuple{Int, Int}, rocks_positions::AbstractVector) = RockPOMDP(map_size=map_size, rocks_positions=rocks_positions)

POMDPs.isterminal(pomdp::RockPOMDP, s::RSState) = s.pos == pomdp.terminal_state.pos 
POMDPs.discount(pomdp::RockPOMDP) = pomdp.discount_factor

#-----------action---------------
const N_BASIC_ACTIONS = 5
const BASIC_ACTIONS_DICT = Dict(:sample => 1,
                                :north => 2, 
                                :east => 3,
                                :south => 4,
                                :west => 5)

const ACTION_DIRS = (RSPos(0,0),
                    RSPos(0,1),
                    RSPos(1,0),
                    RSPos(0,-1),
                    RSPos(-1,0))

POMDPs.actions(pomdp::RockPOMDP{K}) where K = 1:N_BASIC_ACTIONS+K
POMDPs.actionindex(pomdp::RockPOMDP, a::Int) = a

function POMDPs.actions(pomdp::RockPOMDP{K}, s::RSState) where K
    if in(s.pos, pomdp.rocks_positions) # slow? pomdp.rock_pos is a vec 
        return actions(pomdp)
    else
        # sample not available
        return 2:N_BASIC_ACTIONS+K
    end
end
# ------------heuristics--------------
# A fixed action policy which always takes the action `move east`.
struct RSExitSolver <: Solver end
struct RSExit <: Policy
    exit_return::Vector{Float64}
end
POMDPs.solve(::RSExitSolver, m::RockPOMDP) = RSExit([discount(m)^(m.map_size[1]-x) * m.exit_reward for x in 1:m.map_size[1]])
POMDPs.solve(solver::RSExitSolver, m::UnderlyingMDP{P}) where P <: RockPOMDP = solve(solver, m.pomdp)
POMDPs.value(p::RSExit, s::RSState) = s.pos[1] == -1 ? 0.0 : p.exit_return[s.pos[1]]

function POMDPs.value(p::RSExit, b::AbstractParticleBelief)
    utility = 0.0
    for (i, s) in enumerate(particles(b))
        if s.pos[1] != -1 # if s is not terminal
            utility += weight(b, i) * p.exit_return[s.pos[1]]
        end
    end
    return utility / weight_sum(b)
end
POMDPs.action(p::RSExit, b) = 2 # Move east

# Dedicated MDP solver for RockSample
struct RSMDPSolver <: Solver
    include_Q::Bool
end

function POMDPs.solve(solver::RSMDPSolver, m::UnderlyingMDP{P}) where P <: RockPOMDP
    util = rs_mdp_utility(m.pomdp)
    if solver.include_Q
        return solve(ValueIterationSolver(init_util=util, include_Q=true), m)
    else
        return ValueIterationPolicy(m, utility=util, include_Q=false)
    end
end

# Dedicated QMDP solver for RockSample
struct RSQMDPSolver <: Solver end
function POMDPs.solve(::RSQMDPSolver, m::RockPOMDP)
    vi_policy = solve(RSMDPSolver(include_Q=true), m)
    return AlphaVectorPolicy(m, vi_policy.qmat, vi_policy.action_map)
end

# Solve for the optimal utility of RockSample, assuming full observability.
function rs_mdp_utility(m::RockPOMDP{K}) where K
    util = zeros(length(states(m)))
    discounts = discount(m) .^ (0:(m.map_size[1]+m.map_size[2]-2))

    # Rewards for exiting.
    exit_returns = [discounts[m.map_size[1] - x + 1] * m.exit_reward for x in 1:m.map_size[1]]

    # Calculate the optimal utility for states having no good rocks, which is the exit return.
    rocks = falses(K)
    for x in 1:m.map_size[1]
        for y in 1:m.map_size[2]
            util[stateindex(m, RSState(RSPos(x,y), SVector{K,Bool}(rocks)))] = exit_returns[x]
        end
    end

    # The optimal utility of states having k good rocks can be derived from the utility of states having k-1 good rocks:
    # Utility_k = max(ExitReturn, argmax_{r∈GoodRocks}(γ^{Manhattan distance to r}Utility_{k-1}))
    for good_rock_num in 1:K
        for good_rocks in combinations(1:K, good_rock_num)
            rocks = falses(K)
            for good_rock in good_rocks
                rocks[good_rock] = true
            end
            for x in 1:m.map_size[1]
                for y in 1:m.map_size[2]
                    best_return = exit_returns[x]
                    for good_rock in good_rocks
                        dist_to_good_rock = abs(x - m.rocks_positions[good_rock][1]) + abs(y - m.rocks_positions[good_rock][2])
                        rocks[good_rock] = false
                        sample_return = discounts[dist_to_good_rock+1] * (m.good_rock_reward + discounts[2] * util[stateindex(m, RSState(m.rocks_positions[good_rock], SVector{K,Bool}(rocks)))])
                        rocks[good_rock] = true
                        if sample_return > best_return
                            best_return = sample_return
                        end
                    end
                    util[stateindex(m, RSState(RSPos(x,y), SVector{K,Bool}(rocks)))] = best_return
                end
            end
        end
    end

    return util
end

#-----------observations-----------
const OBSERVATION_NAME = (:good, :bad, :none)

POMDPs.observations(pomdp::RockPOMDP) = 1:3
POMDPs.obsindex(pomdp::RockPOMDP, o::Int) = o

function POMDPs.observation(pomdp::RockPOMDP, a::Int, s::RSState)
    if a <= N_BASIC_ACTIONS
        # no obs
        return SparseCat((1,2,3), (0.0,0.0,1.0)) # for type stability
    else
        rock_ind = a - N_BASIC_ACTIONS 
        rock_pos = pomdp.rocks_positions[rock_ind]
        dist = norm(rock_pos - s.pos)
        efficiency = 0.5*(1.0 + exp(-dist*log(2)/pomdp.sensor_efficiency))
        rock_state = s.rocks[rock_ind]
        if rock_state
            return SparseCat((1,2,3), (efficiency, 1.0 - efficiency, 0.0))
        else
            return SparseCat((1,2,3), (1.0 - efficiency, efficiency, 0.0))
        end
    end
end

#-----------------reward----------------
function POMDPs.reward(pomdp::RockPOMDP, s::RSState, a::Int)
    r = pomdp.step_penalty
    if next_position(s, a)[1] > pomdp.map_size[1]
        r += pomdp.exit_reward
        return r
    end

    if a == BASIC_ACTIONS_DICT[:sample] && in(s.pos, pomdp.rocks_positions) # sample 
        rock_ind = findfirst(isequal(s.pos), pomdp.rocks_positions) # slow ?
        r += s.rocks[rock_ind] ? pomdp.good_rock_reward : pomdp.bad_rock_penalty 
    elseif a > N_BASIC_ACTIONS # using senssor
        r += pomdp.sensor_use_penalty
    end
    return r
end
#---------------states----------------
function POMDPs.stateindex(pomdp::RockPOMDP{K}, s::RSState{K}) where K
    if isterminal(pomdp, s)
        return length(pomdp)
    end
    return s.pos[1] + pomdp.indices[1] * (s.pos[2]-1) + dot(view(pomdp.indices, 2:(K+1)), s.rocks)
end

function state_from_index(pomdp::RockPOMDP{K}, si::Int) where K
    if si == length(pomdp)
        return pomdp.terminal_state
    end
    rocks_dim = @SVector fill(2, K)
    nx, ny = pomdp.map_size
    s = CartesianIndices((nx, ny, rocks_dim...))[si]
    pos = RSPos(s[1], s[2])
    rocks = SVector{K, Bool}(s.I[3:(K+2)] .- 1)
    return RSState{K}(pos, rocks)
end

# the state space is the pomdp itself
POMDPs.states(pomdp::RockPOMDP) = pomdp

Base.length(pomdp::RockPOMDP) = pomdp.map_size[1]*pomdp.map_size[2]*2^length(pomdp.rocks_positions) + 1

# we define an iterator over it
function Base.iterate(pomdp::RockPOMDP, i::Int=1)
    if i > length(pomdp)
        return nothing
    end
    s = state_from_index(pomdp, i)
    return (s, i+1)
end

function POMDPs.initialstate(pomdp::RockPOMDP{K}) where K
    probs = normalize!(ones(2^K), 1)
    states = Vector{RSState{K}}(undef, 2^K)
    for (i,rocks) in enumerate(Iterators.product(ntuple(x->[false, true], K)...))
        states[i] = RSState{K}(pomdp.init_pos, SVector(rocks))
    end
    return SparseCat(states, probs)
end

# ----------------transition------------------
function POMDPs.transition(pomdp::RockPOMDP{K}, s::RSState{K}, a::Int) where K
    if isterminal(pomdp, s)
        return Deterministic(pomdp.terminal_state)
    end
    new_pos = next_position(s, a)
    if a == BASIC_ACTIONS_DICT[:sample] && in(s.pos, pomdp.rocks_positions)
        rock_ind = findfirst(isequal(s.pos), pomdp.rocks_positions) # slow ?
        # set the new rock to bad
        new_rocks = MVector{K, Bool}(undef)
        for r=1:K
            new_rocks[r] = r == rock_ind ? false : s.rocks[r]
        end
        new_rocks = SVector(new_rocks)
    else 
        new_rocks = s.rocks
    end

    #=
    if new_pos[1] > pomdp.map_size[1]
        # the robot reached the exit area
        new_pos = RSPos(pomdp.init_pos)
        new_state = RSState{K}(new_pos, new_rocks)
    =#
    if new_pos[1] > pomdp.map_size[1]
        # the robot reached the exit area
        new_state = pomdp.terminal_state
    
    else
        new_pos = RSPos(clamp(new_pos[1], 1, pomdp.map_size[1]), 
                        clamp(new_pos[2], 1, pomdp.map_size[2]))
        new_state = RSState{K}(new_pos, new_rocks)
    end
    return Deterministic(new_state)
end

function next_position(s::RSState, a::Int)
    if a < N_BASIC_ACTIONS
        # the robot moves 
        return s.pos + ACTION_DIRS[a]
    elseif a >= N_BASIC_ACTIONS 
        # robot check rocks or samples
        return s.pos
    else
        throw("ROCKSAMPLE ERROR: action $a not valid")
    end
end
#------------visualization--------------
function POMDPTools.render(pomdp::RockPOMDP, step;
    viz_rock_state=true,
    viz_belief=true,
    pre_act_text=""
)
    nx, ny = pomdp.map_size[1] + 1, pomdp.map_size[2] + 1
    cells = []
    for x in 1:nx-1, y in 1:ny-1
        ctx = cell_ctx((x, y), (nx, ny))
        cell = compose(ctx, rectangle(), fill("white"))
        push!(cells, cell)
    end
    grid = compose(context(), linewidth(0.5mm), stroke("gray"), cells...)
    outline = compose(context(), linewidth(1mm), rectangle())

    rocks = []
    for (i, (rx, ry)) in enumerate(pomdp.rocks_positions)
        ctx = cell_ctx((rx, ry), (nx, ny))
        clr = "black"
        if viz_rock_state && get(step, :s, nothing) !== nothing
            clr = step[:s].rocks[i] ? "green" : "red"
        end
        rock = compose(ctx, ngon(0.5, 0.5, 0.3, 6), stroke(clr), fill("gray"))
        push!(rocks, rock)
    end
    rocks = compose(context(), rocks...)
    exit_area = render_exit((nx, ny))

    agent = nothing
    action = nothing
    if get(step, :s, nothing) !== nothing
        agent_ctx = cell_ctx(step[:s].pos, (nx, ny))
        agent = render_agent(agent_ctx)
        if get(step, :a, nothing) !== nothing
            action = render_action(pomdp, step)
        end
    end
    action_text = render_action_text(pomdp, step, pre_act_text)

    belief = nothing
    if viz_belief && (get(step, :b, nothing) !== nothing)
        belief = render_belief(pomdp, step)
    end
    sz = min(w, h)
    return compose(context((w - sz) / 2, (h - sz) / 2, sz, sz), action, agent, belief,
        exit_area, rocks, action_text, grid, outline)
end

function cell_ctx(xy, size)
    nx, ny = size
    x, y = xy
    return context((x - 1) / nx, (ny - y - 1) / ny, 1 / nx, 1 / ny)
end

function render_belief(pomdp::RockPOMDP, step)
    rock_beliefs = get_rock_beliefs(pomdp, get(step, :b, nothing))
    nx, ny = pomdp.map_size[1] + 1, pomdp.map_size[2] + 1
    belief_outlines = []
    belief_fills = []
    for (i, (rx, ry)) in enumerate(pomdp.rocks_positions)
        ctx = cell_ctx((rx, ry), (nx, ny))
        clr = "black"
        belief_outline = compose(ctx, rectangle(0.1, 0.87, 0.8, 0.07), stroke("gray31"), fill("gray31"))
        belief_fill = compose(ctx, rectangle(0.1, 0.87, rock_beliefs[i] * 0.8, 0.07), stroke("lawngreen"), fill("lawngreen"))
        push!(belief_outlines, belief_outline)
        push!(belief_fills, belief_fill)
    end
    return compose(context(), belief_fills..., belief_outlines...)
end

function get_rock_beliefs(pomdp::RockPOMDP{K}, b) where K
    rock_beliefs = zeros(Float64, K)
    for (sᵢ, bᵢ) in weighted_iterator(b)
        rock_beliefs[sᵢ.rocks.==1] .+= bᵢ
    end
    return rock_beliefs
end

function render_exit(size)
    nx, ny = size
    x = nx
    y = ny
    ctx = context((x - 1) / nx, (ny - y) / ny, 1 / nx, 1)
    rot = Rotation(pi / 2, 0.5, 0.5)
    txt = compose(ctx, text(0.5, 0.5, "EXIT AREA", hcenter, vtop, rot),
        stroke("black"),
        fill("black"),
        fontsize(20pt))
    return compose(ctx, txt, rectangle(), fill("red"))
end

function render_agent(ctx)
    center = compose(context(), circle(0.5, 0.5, 0.3), fill("orange"), stroke("black"))
    lwheel = compose(context(), ellipse(0.2, 0.5, 0.1, 0.3), fill("orange"), stroke("black"))
    rwheel = compose(context(), ellipse(0.8, 0.5, 0.1, 0.3), fill("orange"), stroke("black"))
    return compose(ctx, center, lwheel, rwheel)
end

function render_action_text(pomdp::RockPOMDP, step, pre_act_text)
    actions = ["Sample", "North", "East", "South", "West"]
    action_text = "Terminal"
    if get(step, :a, nothing) !== nothing
        if step.a <= N_BASIC_ACTIONS
            action_text = actions[step.a]
        else
            action_text = "Sensing Rock $(step.a - N_BASIC_ACTIONS)"
        end
    end
    action_text = pre_act_text * action_text

    _, ny = pomdp.map_size
    ny += 1
    ctx = context(0, (ny - 1) / ny, 1, 1 / ny)
    txt = compose(ctx, text(0.5, 0.5, action_text, hcenter),
        stroke("black"),
        fill("black"),
        fontsize(20pt))
    return compose(ctx, txt, rectangle(), fill("white"))
end

function render_action(pomdp::RockPOMDP, step)
    if step.a == BASIC_ACTIONS_DICT[:sample]
        ctx = cell_ctx(step.s.pos, pomdp.map_size .+ (1, 1))
        if in(step.s.pos, pomdp.rocks_positions)
            rock_ind = findfirst(isequal(step.s.pos), pomdp.rocks_positions)
            clr = step.s.rocks[rock_ind] ? "green" : "red"
        else
            clr = "black"
        end
        return compose(ctx, ngon(0.5, 0.5, 0.1, 6), stroke("gray"), fill(clr))
    elseif step.a > N_BASIC_ACTIONS
        rock_ind = step.a - N_BASIC_ACTIONS
        rock_pos = pomdp.rocks_positions[rock_ind]
        nx, ny = pomdp.map_size[1] + 1, pomdp.map_size[2] + 1
        rock_pos = ((rock_pos[1] - 0.5) / nx, (ny - rock_pos[2] - 0.5) / ny)
        rob_pos = ((step.s.pos[1] - 0.5) / nx, (ny - step.s.pos[2] - 0.5) / ny)
        sz = min(w, h)
        return compose(context((w - sz) / 2, (h - sz) / 2, sz, sz), line([rob_pos, rock_pos]), stroke("orange"), linewidth(0.01w))
    end
    return nothing
end

#------------------------model checking--------------------------
function POMDPModelChecking.labels(pomdp::RockPOMDP, s::RSState, a::Int64)
    if a == RockSample.BASIC_ACTIONS_DICT[:sample] && in(s.pos, pomdp.rocks_positions) # sample
        rock_ind = findfirst(isequal(s.pos), pomdp.rocks_positions) # slow ?
        if s.rocks[rock_ind]
            return (:good_rock,)
        else
            return (:bad_rock,)
        end
    end
    if isterminal(pomdp, s)
        return (:exit,)
    end
    return ()
end