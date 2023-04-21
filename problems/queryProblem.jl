using POMDPs

# Define a state type
struct QueryState
    status::Bool
end

# Define an action type
struct QueryAction
end

# Define an observation type
struct QueryObservation
    result::Bool
end

struct queryPOMDP <: POMDP{Bool, String, String}
    POMDP{QueryState, QueryAction, QueryObservation}


    p_correct::Float64
    indices::Dict{String, Int}

    TigerPOMDP(p_correct=0.85) = new(p_correct, Dict("left"=>1, "right"=>2, "listen"=>3))
end

# Define a transition function
function POMDPs.transition(prob::NamedTuple, state::QueryState, action::QueryAction)
    if state.status
        # If the query has already been executed, the state does not change
        return state
    else
        # If the query has not been executed, there is a 50-50 chance of success or failure
        new_status = rand(Bool)
        return QueryState(new_status)
    end
end

# Define an observation function
function POMDPs.observation(prob::NamedTuple, state::QueryState, action::QueryAction, next_state::QueryState)
    if next_state.status
        # If the query was successful, the observation is accurate
        return QueryObservation(true)
    else
        # If the query failed, the observation is inaccurate with 20% chance of being correct
        correct = rand() < 0.2
        return QueryObservation(correct)
    end
end

# Define a reward function
function POMDPs.reward(prob::NamedTuple, state::QueryState, action::QueryAction, next_state::QueryState)
    if next_state.status
        # If the query was successful, the reward is 10
        return 10.0
    else
        # If the query failed, the reward is -1
        return -1.0
    end
end

# Define a discount factor
discount_factor = 0.95

# Define a POMDP problem
query_problem = POMDP{QueryState, QueryAction, QueryObservation}(transition, observation, reward, discount_factor)
