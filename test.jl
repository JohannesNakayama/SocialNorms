# imports
using Random
using Statistics
using PyPlot

mutable struct CircularBitArray
    array::BitArray
    function CircularBitArray(b::BitArray, rng=Random.MersenneTwister(0))
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
        noise=0.0,
        rng=Random.MersenneTwister(0)
    )
        new(
            circ_bit_arr,
            Random.rand!(rng, zeros(length(circ_bit_arr)), 1:max_radius),
            tolerance,
            sim_iterations,
            noise
        )
    end

end

function compute_norm(agent_list::CircularBitArray, agent::Int64, radius::Int64)
    return Statistics.mean(get_k_neighborhood(agent_list, agent, radius))
end

function update_radius!(radius_list, agent_list, current_agent, tolerance)

    # current properties
    current_radius = radius_list[current_agent]
    current_norm = agent_list.array[current_agent]

    # update radius
    if (
        (abs(compute_norm(agent_list, current_agent, current_radius)
            - compute_norm(agent_list, current_agent, current_radius + 1))
        > tolerance)
        & (current_radius < ((length(agent_list) / 2) - 1))
    )
        radius_list[current_agent] += 1
        print("INCREASED")
    elseif (
        (abs(compute_norm(agent_list, current_agent, current_radius)
            - compute_norm(agent_list, current_agent, current_radius - 1))
        < tolerance)
        & (current_radius > 1)
    )
        radius_list[current_agent] -= 1
        print("DECREASED")
    end

    return radius_list

end

function update_norm!(agent_list, radius_list, current_agent, noise=0.0, rng=Random.MersenneTwister(0))

    current_radius = radius_list[current_agent]
    current_norm = agent_list.array[current_agent]
    computed_norm = compute_norm(agent_list, current_agent, current_radius)

    # update norm regularly
    if computed_norm > 0.5
        agent_list.array[current_agent] = true
    elseif computed_norm < 0.5
        agent_list.array[current_agent] = false
    end

    # introduce some noise
    if rand(rng, Float64) < noise
        agent_list.array[current_agent] = Random.bitrand(rng, 1)[1]
    end

    return agent_list

end

function tick!(state, config, rng=Random.MersenneTwister(0))

    for i in 1:length(state[1])
        adapting_agent = rand(rng, 1:length(state[1]))
        update_radius!(state[2], state[1], adapting_agent, config.tolerance)
        update_norm!(state[1], state[2], adapting_agent, config.noise, rng)
    end

    return state

end

function simulate(config; display=true)

    agent_list = deepcopy(config.agent_list)
    radius_list = deepcopy(config.radius_list)
    state = (agent_list, radius_list)

    norm_matrix = zeros(Int64, config.sim_iterations + 1, length(state[1]))
    radius_matrix = zeros(Int64, config.sim_iterations + 1, length(state[1]))

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
Random.seed!(7)
cfg1 = Config(
    CircularBitArray(191),
    20,
    0.05,
    400,
    0.0
)
simulate(cfg1)

Random.seed!(2)
cfg2 = Config(
    CircularBitArray(BitArray(zeros(191))),
    60,
    0.05,
    400,
    0.0
)
res2 = simulate(cfg2)
