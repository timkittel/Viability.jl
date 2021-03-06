

module Viability

using Base
using NearestNeighbors

# type definitions
COORDINATE_TYPE = Float64
POINT_TYPE = Array{COORDINATE_TYPE,1}
POINTS_ARRAY_TYPE = Array{COORDINATE_TYPE,2}
STATE_TYPE = UInt8
STATE_ARRAY_TYPE = Array{STATE_TYPE,1}
STATE_ARRAY_ARGUMENT_TYPE = AbstractArray{STATE_TYPE,1}
SHARED_STATE_ARRAY_TYPE = SharedArray{STATE_TYPE,1}

@enum SP_COMPUTATION_TYPE sp_parallel sp_mparallel

# computational globals
MAX_SP_VIABILITY_ITERATION_STEPS = 100

RequiredArgument(name) = error("$name is a required keyword argument")
RequiredArgument(name, explanation) = error("$name is a required keyword argument. $explanation")

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
end

function spatial_compactification(
        f,
        xi_mid::COORDINATE_TYPE,
        axis::Int
    )
    function f_compactified(y)
        x = copy(y)
        x[axis] = y[axis] * xi_mid / (1-y[axis])
        p = copy(y)
        p[axis] = (1-y[axis])^2 / xi_mid
        p*f(x)
    end
end

function spatial_compactification(
    f,
    x_mid::POINT_TYPE,
    axes::Array{Int,1}
    )
    @assert len(x_mid) == len(axes)
    for (xi_mid, axis) in zip(xs_mid, axes)
        f = spatial_compactification(f, xi_mid, axis)
    end
    f
end


function rhs2stepfct(rhs, delta_t)
    function stepfct(x)
        x + rhs(x)*delta_t
    end
    return stepfct
end


function get_indices_of_neighbors_in_ball(
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

function intersects(a::Array, b::Array)
    for i=1:size(a, 1), j=1:size(b, 1)
        if a[i] == b[j]
            return true
        end
    end
    return false
end

function sp_viability_kernel(
        points::POINTS_ARRAY_TYPE,
        states::STATE_ARRAY_ARGUMENT_TYPE,
        step_functions::Array;
        good_states::STATE_ARRAY_ARGUMENT_TYPE=STATE_TYPE[1],
        bad_states::STATE_ARRAY_ARGUMENT_TYPE=STATE_TYPE[0],
        computation_mode::SP_COMPUTATION_TYPE=sp_mparallel,
        delta::COORDINATE_TYPE=RequiredArgument("delta"),
        r::COORDINATE_TYPE=0.0,
        leafsize=10
    )
    ret_val = 2

    # create a kdtree that is used for the find the surrounding indices during the computation
    kdtree = KDTree(points; leafsize=leafsize)
    # create a shared array copy of states for parallelization
    inp_states = states
    states = convert(SharedArray, states)
    if computation_mode == sp_parallel
        ret_val = _sp_viability_kernel_parallel(points, states, step_functions, kdtree,
            good_states=good_states,
            unsuccessful_states=bad_states,
            successful_states=good_states,
            work_states=good_states,
            delta=delta
        )
    elseif computation_mode == sp_mparallel
        if r == 0.0
            RequiredArgument(
                "r",
                "If unsure, choose r to be the (maximal) step length of the step_function.")
        end
        ret_val = _sp_viability_kernel_mparallel(points, states, step_functions, kdtree,
            good_states=good_states,
            unsuccessful_states=bad_states,
            successful_states=good_states,
            work_states=good_states,
            delta=delta,
            r=r
        )
    else
        throw(ArgumentError("unknown computation_mode=$computation_mode"))
    end
    # copy back from shared array
    inp_states[:] = states
    return ret_val
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
    _sp_viability_kernel_parallel(points, states, step_functions,
        good_states=target_states+reached_states,
        work_states=work_states,
        successful_states=reached_states,
        unsuccessful_states=work_states,
        delta=delta
    )
end

function _sp_viability_kernel_mparallel(
        points::POINTS_ARRAY_TYPE,
        states::STATE_ARRAY_ARGUMENT_TYPE,
        step_functions::Array,
        kdtree::NearestNeighbors.KDTree;
        good_states::STATE_ARRAY_ARGUMENT_TYPE=RequiredArgument("good_states"),
        work_states::STATE_ARRAY_ARGUMENT_TYPE=RequiredArgument("work_states"),
        successful_states::STATE_ARRAY_ARGUMENT_TYPE=RequiredArgument("successful_states"),
        unsuccessful_states::STATE_ARRAY_ARGUMENT_TYPE=RequiredArgument("unsuccessful_states"),
        delta::COORDINATE_TYPE=RequiredArgument("delta"),
        r::COORDINATE_TYPE=RequiredArgument("r")
    )
    # check the input for correctness and consistency
    @assert ndims(points) == 2
    @assert ndims(states) == 1
    dim = size(points,1)
    n = size(points,2)
    @assert size(states, 1) == n
    @assert size(successful_states) == size(work_states)
    @assert size(unsuccessful_states) == size(work_states)

    # compute r_min and r_max
    r_min = r - delta - 0.01
    # TODO: check whether r_min really makes sense
    r_max = r + delta + 0.01
    println("r_min $r_min")
    println("r_max $r_max")
    # create an array where the points are marked if they have to be checked
    # in the next run
    # initialize all to true, because in the beginning, all points have to be
    # visited at least once
    marked_points = SharedArray{Bool,1}(size(states))
    marked_points[:] = true
    retval = 1

    for iteration_step = 1:MAX_SP_VIABILITY_ITERATION_STEPS
        # get the indices of all points that need to be visited in this iteration
        run_indices = find(marked_points)
        if isempty(run_indices)
            println("yay, I am done")
            retval = 0
            break
        end
        marked_points[:] = false
        println("start iteration $iteration_step")
        @sync @parallel for i in run_indices
            # check whether the state of i is a work state and if yes find the index
            work_state_index = findfirst(work_states, states[i])
            if work_state_index != 0
                # yes, it is a work state

                old_state = states[i] # save it for the comparison later
                # evaluate which new state
                states[i] = _sp_get_new_state(
                    work_state_index, points[:, i], states, step_functions, kdtree,
                    good_states=good_states,
                    successful_states=successful_states,
                    unsuccessful_states=unsuccessful_states,
                    delta=delta
                )
                if old_state != states[i] # did something change?)
                    # mark all points with a distance between r_min and r_max,
                    # because these are the only ones that could reach points[:, i]
                    surrounding_indices = inrange(kdtree, points[:, i], r_max, false)
                    distances = map( index->norm(points[:, index] - points[:, i]), surrounding_indices )
                    for (new_index, new_distance) in zip(surrounding_indices, distances)
                        if !(marked_points[new_index]) && # it's not marked already
                                new_distance > r_min && # distance between r_min and r_max
                                states[new_index] in work_states # it's a work state
                            # mark point for revisiting if it can reach 'point'
                            marked_points[new_index] = true
                        end
                    end
                end
            end
        end
    end
    return retval
end

function _sp_viability_kernel_parallel(
        points::POINTS_ARRAY_TYPE,
        states::STATE_ARRAY_ARGUMENT_TYPE,
        step_functions::Array,
        kdtree::NearestNeighbors.KDTree;
        good_states::STATE_ARRAY_ARGUMENT_TYPE=RequiredArgument("good_states"),
        work_states::STATE_ARRAY_ARGUMENT_TYPE=RequiredArgument("work_states"),
        successful_states::STATE_ARRAY_ARGUMENT_TYPE=RequiredArgument("successful_states"),
        unsuccessful_states::STATE_ARRAY_ARGUMENT_TYPE=RequiredArgument("unsuccessful_states"),
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

    # changed = true
    retval = 1 # unsuccesful
    for it = 1:MAX_SP_VIABILITY_ITERATION_STEPS
        println("start iteration $it")
        changed =  @parallel (|) for i = 1:n
            # check whether the state of i is a work state and if yes find the index
            work_state_index = findfirst(work_states, states[i])
            if work_state_index != 0
                # yes, it is a work state

                old_state = states[i] # save it for the comparison later
                # evaluate which new state
                states[i] = _sp_get_new_state(
                    work_state_index, points[:, i], states, step_functions, kdtree,
                    good_states=good_states,
                    successful_states=successful_states,
                    unsuccessful_states=unsuccessful_states,
                    delta=delta
                )
                old_state != states[i] # did something change?
            else
                false # return that nothing changed
            end
        end
        if !changed # nothing changed? yay, I am done
            println("nothing changed, yay I am done")
            retval = 0 # succesfully finished
            break
        end
    end
    return retval
end

@inline function _sp_get_new_state(
    work_state_index::Int64,
    point::POINT_TYPE,
    states::AbstractArray{STATE_TYPE,1},
    step_functions::Array,
    kdtree::NearestNeighbors.KDTree;
    good_states::STATE_ARRAY_TYPE=RequiredArgument("good_states"),
    successful_states::STATE_ARRAY_TYPE=RequiredArgument("successful_states"),
    unsuccessful_states::STATE_ARRAY_TYPE=RequiredArgument("unsuccessful_states"),
    delta::COORDINATE_TYPE=RequiredArgument("delta")
    )
    for sf in step_functions
        next_point = sf(point)
        surrounding_indices = inrange(kdtree, next_point, delta, false)
        if isempty(surrounding_indices)
            # take the closest point if there was none found in the delta-neighborhood of next_point
            surrounding_indices = knn(kdtree, next_point, 1)[1]
        end
        if intersects(states[surrounding_indices], good_states)
            found_good_state = true
            return successful_states[work_state_index]
        end
    end
    # no good state found, so return the unsuccesful state
    unsuccessful_states[work_state_index]
end

STATE_IDENTIFICATION_VIABILITY_KERNEL = Dict{String,STATE_TYPE}(
    "good" => 1,
    "bad" => 0
)

STATE_IDENTIFICATION_TSM = Dict{String,STATE_TYPE}(
    "unknown" => 0,
    "shelter" => 1,
    "glade" => 2,
    "lake" => 3,
    "remaining sunny upstream" => 4,
    "dark upstream" => 5,
    "backwaters" => 6,
    "remaining sunny downstream" => 7,
    "dark downstream" => 8,
    "sunny eddies" => 9,
    "dark eddies" => 10,
    "sunny abyss" => 11,
    "dark abyss" => 12,
    "trench" => 13
)

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

end
