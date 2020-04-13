# imports
using Random
using Statistics
using PyPlot

mutable struct CircularBitArray
    array::BitArray
    function CircularBitArray(b::BitArray)
        new(b)
    end
end

CircularBitArray(n::Int64) = CircularBitArray(Random.bitrand(n))

Base.:length(c::CircularBitArray) = Base.length(c.array)

function get_k_neighborhood(circ_bit_arr::CircularBitArray, index, k)

    # catch more than entire array case
    if k > (length(circ_bit_arr) / 2) - 1
        error("Subsetting with the chosen k exceeds the length of the CircularBitArray.")
    end

    if index - k < 1  # underflow
        lower_bound_low = 1
        upper_bound_low = index + k
        lower_bound_high = length(circ_bit_arr) - abs(index - k)
        upper_bound_high = length(circ_bit_arr)
        return vcat(
            deepcopy(circ_bit_arr.array[lower_bound_high:upper_bound_high]),
            deepcopy(circ_bit_arr.array[lower_bound_low:upper_bound_low])
        )
    elseif index + k > length(circ_bit_arr)  # overflow
        lower_bound_low = 1
        upper_bound_low = abs(index + k) - length(circ_bit_arr)
        lower_bound_high = index - k
        upper_bound_high = length(circ_bit_arr)
        return vcat(
            deepcopy(circ_bit_arr.array[lower_bound_high:upper_bound_high]),
            deepcopy(circ_bit_arr.array[lower_bound_low:upper_bound_low])
        )
    else  # regular case
        return circ_bit_arr.array[index - k:index + k]
    end

end

function compute_norm(agent_list::CircularBitArray, agent::Int64, radius::Int64)
    return Statistics.mean(get_k_neighborhood(agent_list, agent, radius))
end

function update_radius!(radius_list, agent_list, current_agent, tolerance)

    # current properties
    current_radius = radius_list[current_agent]
    current_norm = agent_list.array[current_agent]

    if (current_radius < (length(agent_list) \ 2) - 1 & current_radius > 1)


        # update radius
        if (
            abs(compute_norm(agent_list, current_agent, current_radius)
            - compute_norm(agent_list, current_agent, current_radius + 1)) > tolerance
        )
            radius_list[current_agent] += 1
        elseif (
            abs(compute_norm(agent_list, current_agent, current_radius)
            - compute_norm(agent_list, current_agent, current_radius - 1)) < tolerance
        )
            radius_list[current_agent] -= 1
        end

    end

    return radius_list

end

function update_norm!(agent_list, radius_list, current_agent, noise=0.0)

    current_radius = radius_list[current_agent]
    current_norm = agent_list.array[current_agent]

    computed_norm = compute_norm(agent_list, current_agent, current_radius)

    if computed_norm > 0.5
        agent_list.array[current_agent] = true
    elseif computed_norm < 0.5
        agent_list.array[current_agent] = false
    end

    if Random.rand() < noise
        agent_list.array[current_agent] = Random.bitrand(1)[1]
    end

    return agent_list

end

mutable struct Config

    agent_list::CircularBitArray
    radius_list::Array{Int64, 1}
    tolerance::Float64
    sim_iterations::Int64
    noise::Float64

    function Config(
        circ_bit_arr::CircularBitArray,
        max_radius,
        tolerance,
        sim_iterations,
        noise=0.0
    )
        new(
            circ_bit_arr,
            Random.rand(1:max_radius, length(circ_bit_arr)),
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

    state = (deepcopy(config.agent_list), deepcopy(config.radius_list))
    norm_matrix = zeros(Int64, config.sim_iterations + 1, length(config.agent_list))
    radius_matrix = zeros(Int64, config.sim_iterations + 1, length(config.agent_list))

    # add initial state to matrices
    norm_matrix[1, :] = state[1].array
    radius_matrix[1, :] = state[2]

    # simulation
    for i in 1:config.sim_iterations
        new_state = tick!(state, config)
        norm_matrix[i + 1, :] = deepcopy(new_state[1].array)
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
cfg = Config(
    CircularBitArray(50),
    15,
    0.05,
    10,
    0.8
)
cfg2 = Config(
    CircularBitArray(BitArray(zeros(50))),
    15,
    0.05,
    100,
    0.8
)
results = simulate(cfg2)

# TO DO: debug update_radius!() 
