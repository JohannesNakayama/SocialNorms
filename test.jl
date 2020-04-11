# imports
using Random
using PyPlot

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

function update_norm!(agent_list, radius_list, current_agent, noise=0.0)

    current_radius = radius_list[current_agent]
    current_norm = agent_list[current_agent]

    computed_norm = compute_norm(agent_list, current_agent, current_radius, current_norm)

    if computed_norm > 0.5
        agent_list[current_agent] = true
    elseif computed_norm < 0.5
        agent_list[current_agent] = false
    end

    if Random.rand() < noise
        agent_list[current_agent] = Random.bitrand(1)[1]
    end

    return agent_list

end

mutable struct Config

    agent_list::BitArray
    radius_list::Array{Int64, 1}
    tolerance::Float64
    sim_iterations::Int64
    noise::Float64

    function Config(agent_count, max_radius, tolerance, sim_iterations, noise=0.0)
        new(
            Random.bitrand(agent_count),
            Random.rand(1:max_radius, agent_count),
            tolerance,
            sim_iterations,
            noise
        )
    end

end

function tick!(state, config)

    for i in 1:length(state[1])
        adapting_agent = Random.rand(1:length(state[1]))
        update_radius!(state[2], state[1], adapting_agent, config.tolerance)
        update_norm!(state[1], state[2], adapting_agent)
    end

    return state

end

function simulate(config; display=true)

    state = (config.agent_list, config.radius_list)
    norm_matrix = zeros(Int64, config.sim_iterations + 1, length(config.agent_list))
    radius_matrix = zeros(Int64, config.sim_iterations + 1, length(config.agent_list))

    # add initial state to matrices
    norm_matrix[1, :] = state[1]
    radius_matrix[1, :] = state[2]

    # simulation
    for i in 1:config.sim_iterations
        new_state = tick!(state, config)
        norm_matrix[i + 1, :] = deepcopy(new_state[1])
        radius_matrix[i + 1, :] = deepcopy(new_state[2])
    end
    results = (norm_matrix, radius_matrix)

    # visualize results
    if display
        display_results(results)
    end

    return results

end

function display_results(results)

    PyPlot.clf()
    PyPlot.subplot(1, 2, 1)
    PyPlot.tick_params(
        axis="both",          # changes apply to the x-axis
        which="both",      # both major and minor ticks are affected
        bottom=false,      # ticks along the bottom edge are off
        top=false,         # ticks along the top edge are off
        labelbottom=false
    )
    PyPlot.imshow(results[1])
    PyPlot.subplot(1, 2, 2)
    PyPlot.tick_params(
        axis="both",          # changes apply to the x-axis
        which="both",      # both major and minor ticks are affected
        bottom=false,      # ticks along the bottom edge are off
        top=false,         # ticks along the top edge are off
        labelbottom=false
    )
    PyPlot.imshow(results[2])
    display(gcf())

end

# simulation runs
cfg = Config(191, 30, 0.001, 200, 1.0)
results = simulate(cfg)
