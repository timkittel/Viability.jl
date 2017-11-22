
using Base

# type definitions
@everywhere COORDINATE_TYPE = Float64
@everywhere POINT_TYPE = Array{COORDINATE_TYPE,1}
@everywhere POINTS_ARRAY_TYPE = Array{COORDINATE_TYPE,2} # SharedArray
STATE_TYPE = UInt8
STATE_ARRAY_TYPE = Array{STATE_TYPE}

# computational globals
MAX_SP_VIABILITY_ITERATION_STEPS = 100

RequiredArgument(name) = error("$name is a required keyword argument")

function map_over_points{T}(::Type{T}, f, points::POINTS_ARRAY_TYPE)
    @assert ndims(points) == 2
    npoints = size(points, 2)
    res = Array{T}(npoints)
    for i = 1:npoints
        res[i] = f(points[:, i])
    end
    return res
end

function create_normalized_2d_grid(
    n_per_dim::Unsigned;
    epsilon::COORDINATE_TYPE=0.0 # how much space should be left around the boundary
    )
    create_2d_grid(
        n_per_dim,
        COORDINATE_TYPE[0 1; 0 1],
        epsilon=epsilon
    )
end


function create_2d_grid(
    n_per_dim::Unsigned,
    bounds::Array{COORDINATE_TYPE,2};
    epsilon::COORDINATE_TYPE=0.0 # how much space should be left around the boundary
    )
    dim = 2
    @assert ndims(bounds) == 2
    @assert size(bounds, 1) == dim
    @assert size(bounds, 2) == 2
    @assert epsilon >= 0
    grid = POINTS_ARRAY_TYPE(dim, n_per_dim^dim)
    start_points = bounds[:,1] + epsilon
    steps = (bounds[:,2] - bounds[:,1] - 2*epsilon) / (n_per_dim - 1)
    @assert all(steps .> 0.0) "check you bounds and epsilon values"
    for i = 1:n_per_dim, j = 1:n_per_dim
        index  = i + (j-1)*n_per_dim
        grid[1, index] = start_points[1] + steps[1] * (i-1)
        grid[2, index] = start_points[2] + steps[2] * (j-1)
    end
    return grid
end

function point2bounds(
    point::POINT_TYPE,
    bounds::Array{COORDINATE_TYPE,2};
    epsilon::COORDINATE_TYPE=0.0
    )
    offsets = bounds[:,1] + epsilon
    diffs = bounds[:,2] - bounds[:,1] - 2*epsilon
    point.*diffs + offsets
end

function points2bounds(
    points::POINTS_ARRAY_TYPE,
    bounds::Array{COORDINATE_TYPE,2};
    epsilon::COORDINATE_TYPE=0.0
    )
    new_points = zeros(points)
    n = size(points,2)
    for i=1:n
        new_points[:, i] = point2bounds(points[:, i], bounds, epsilon=epsilon)
    end
    return new_points
end

function temporal_homogenization(f; epsilon::COORDINATE_TYPE=1e-6)
    function f_homogenized(x::POINT_TYPE)
        xdot = f(x)
        xdot / (norm(xdot) + epsilon)
    end
end

function spatial_normalization(
        f,
        bounds::Array{COORDINATE_TYPE,2};
        epsilon::COORDINATE_TYPE=0.0 # how much space should be left around the boundary
    )
    offsets = bounds[:,1] + epsilon
    diffs = bounds[:,2] - bounds[:,1] - 2*epsilon
    # println(offsets)
    # println(diffs)
    function f_normalized(y::POINT_TYPE)
        f(y.*diffs + offsets)./diffs
    end
    # println(methods(f_normalized))
end

function rhs2stepfct(rhs, delta_t)
    function stepfct(x)
        x + rhs(x)*delta_t
    end
    return stepfct
end


@everywhere function get_indices_of_neighbors_in_ball(
    point::POINT_TYPE,
    points::POINTS_ARRAY_TYPE,
    epsilon::COORDINATE_TYPE # radius of the ball
    )
    @assert ndims(point) == 1
    dim = size(point, 1)
    @assert ndims(points) == 2
    @assert size(points, 1) == dim
    @assert epsilon > 0
    point_diffs = points .- point
    diffs = sum(point_diffs.^2, [1])
    neighbor_indices = find(diffs .< epsilon^2)
end

@everywhere function intersects(a::Array, b::Array)
    for i=1:size(a, 1), j=1:size(b, 1)
        if a[i] == b[j]
            return true
        end
    end
    return false
end

function sp_viability_kernel(
        points::POINTS_ARRAY_TYPE,
        states::STATE_ARRAY_TYPE,
        step_functions::Array;
        good_states::STATE_ARRAY_TYPE=STATE_TYPE[1],
        bad_states::STATE_ARRAY_TYPE=STATE_TYPE[0],
        delta::COORDINATE_TYPE=RequiredArgument("delta")
    )
    _sp_viability_kernel(points, states, step_functions,
        good_states=good_states,
        unsuccessful_states=bad_states,
        successful_states=good_states,
        work_states=good_states,
        delta=delta
    )
end

function sp_capture_basin(
        points::POINTS_ARRAY_TYPE,
        states::STATE_ARRAY_TYPE,
        step_functions::Array;
        target_states::STATE_ARRAY_TYPE=STATE_TYPE[1],
        work_states::STATE_ARRAY_TYPE=STATE_TYPE[0],
        reached_states::STATE_ARRAY_TYPE=STATE_TYPE[1],
        delta::COORDINATE_TYPE=RequiredArgument("delta")
    )
    _sp_viability_kernel(points, states, step_functions,
        good_states=target_states+reached_states,
        work_states=work_states,
        successful_states=reached_states,
        unsuccessful_states=work_states,
        delta=delta
    )
end

function _sp_viability_kernel(
        points::POINTS_ARRAY_TYPE,
        states::STATE_ARRAY_TYPE,
        step_functions::Array;
        good_states::STATE_ARRAY_TYPE=RequiredArgument("good_states"),
        work_states::STATE_ARRAY_TYPE=RequiredArgument("work_states"),
        successful_states::STATE_ARRAY_TYPE=RequiredArgument("successful_states"),
        unsuccessful_states::STATE_ARRAY_TYPE=RequiredArgument("unsuccessful_states"),
        delta::COORDINATE_TYPE=RequiredArgument("delta")
    )
    # check the input for correctness and consistency
    @assert ndims(points) == 2
    @assert ndims(states) == 1
    dim = size(points,1)
    n = size(points,2)
    @assert size(states, 1) == n
    @assert size(successful_states) == size(work_states)
    @assert size(unsuccessful_states) == size(work_states)

    changed = true
    retval = 1 # unsuccesful
    for it = 1:MAX_SP_VIABILITY_ITERATION_STEPS
        println("start iteration $it")
        changed = false
        # for i = 1:n
        for i = 1:n
            # check whether the state of i is a work state and if yes find the index
            work_state_index = findfirst(work_states, states[i])
            ####################################
            point = points[:, i]
            debugging = false
            if point[1] < 0.03 && point[2] > 0.97
                println("DEBUGGIN MODE point $point")
                debugging = true
            end
            ####################################
            if work_state_index != 0
                # yes, it is a work state

                old_state = states[i] # save it for the comparison later
                found_good_state = false
                for sf in step_functions
                    next_point = sf(points[:, i])
                    surrounding_indices = get_indices_of_neighbors_in_ball(next_point, points, delta)
                    if debugging
                        println("sf $sf")
                        println("next_point $next_point")
                        s = states[surrounding_indices]
                        println("surrounding states $s")
                        ps = points[:, surrounding_indices]
                        println("surround points $ps")
                    end
                    if intersects(states[surrounding_indices], good_states)
                        found_good_state = true
                        states[i] = successful_states[work_state_index]
                        break
                    end
                end
                if !found_good_state
                    states[i] = unsuccessful_states[work_state_index]
                end
                if debugging
                    # error("interrupt")
                end
                # check whether anything changed
                if old_state != states[i]
                    changed = true
                end
            end
            true
        end
        println("stop iteration $it")
        if !changed # nothing changed? yay, I am done
            println("nothing changed, yay I am done")
            retval = 0 # succesfully finished
            break
        end
        true
    end
    return retval
end

function write_result_file(fname, model_info, points, states, delim="; ")
    # check the input for correctness and consistency
    @assert ndims(points) == 2
    @assert ndims(states) == 1
    n = size(points,2)
    @assert size(states, 1) == n
    # TODO: added tests for model_info
    open(fname, "w") do f
        write(f, "#file-version:0.1\n")
        write(f, "#$model_info\n")
        for i = 1:n
            point = points[:, i]
            state = states[i]
            write(f, "$point$delim$state\n")
        end
    end
end
