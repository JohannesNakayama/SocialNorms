using Random

function compute_norm(agent_list, agent, radius, norm)

    # determine base neighborhood
    lower_bound = (agent - radius) > 0 ? (agent - radius) : 1
    upper_bound = (agent + radius) > length(agent_list) ? length(agent_list) : agent + radius
    base_neighborhood = deepcopy(agent_list[lower_bound:upper_bound])

    # account for potential overflows
    lower_tail = agent_list[1:(agent + radius - length(agent_list))]
    upper_tail = agent_list[((length(agent_list) + 1) - abs(agent - radius)):length(agent_list)]

    # concatenate all fragments
    neighborhood = vcat(base_neighborhood, lower_tail, upper_tail)

    # tie breaking mechanism (keep current norm in case of a tie
    return (sum(neighborhood) / length(neighborhood))

end

function update_radius!(radius_list, agent_list, current_agent, tolerance)

    # current properties
    current_radius = radius_list[current_agent]
    current_norm = agent_list[current_agent]

    # update radius
    if abs(compute_norm(agent_list, current_agent, current_radius, current_norm)
        - compute_norm(agent_list, current_agent, current_radius + 1, current_norm)) > tolerance

        if current_radius < length(agent_list)
            radius_list[current_agent] += 1
        end

    elseif abs(compute_norm(agent_list, current_agent, current_radius, current_norm)
        - compute_norm(agent_list, current_agent, current_radius - 1, current_norm)) < tolerance

        if current_radius > 1
            radius_list[current_agent] -= 1
        end

    end

    return radius_list

end

function update_norm!(agent_list, radius_list, current_agent)

    current_radius = radius_list[current_agent]
    current_norm = agent_list[current_agent]

    computed_norm = compute_norm(agent_list, current_agent, current_radius, current_norm)

    if computed_norm > 0.5
        agent_list[current_agent] = true
    elseif computed_norm < 0.5
        agent_list[current_agent] = false
    end

    return agent_list

end

mutable struct Config

    agent_list::BitArray
    radius_list::Array{Int64, 1}
    tolerance::Float64
    sim_iterations::Int64

    function Config(agent_count, max_radius, tolerance, sim_iterations)
        new(
            bitrand(agent_count),
            rand(1:max_radius, agent_count),
            tolerance,
            sim_iterations
        )
    end

end

function tick!(state, config)

    for i in 1:length(state[1])
        adapting_agent = rand(1:length(state[1]))
        update_radius!(state[2], state[1], adapting_agent, config.tolerance)
        update_norm!(state[1], state[2], adapting_agent)
    end

    return state

end



cfg = Config(100, 30, 0.05, 100)
state = (cfg.agent_list, cfg.radius_list)

tick!(state, cfg)
