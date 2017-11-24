
# start up kernels if not already done
# if Sys.CPU_CORES > nworkers()
#     if nworkers() == 1
#         addprocs(Sys.CPU_CORES)
#     else
#         addprocs(Sys.CPU_CORES - nworkers())
#     end
# end

include("Viability.jl")

OUTFILENAME = "consum.txt"

dim = 2
num_per_dim = unsigned(401)
stepsize=1/(num_per_dim-1)
println("stepsize $stepsize")
delta_t = 7*stepsize
println("delta_t $delta_t")
delta_ball = 1.5*stepsize
println("delta_ball $delta_ball")
num_total = num_per_dim ^ 2
bounds = COORDINATE_TYPE[0 2; 0 3]

estimation_args = Dict{String,Any}(
    "num-per-dim" => num_per_dim,
    "bounds" => bounds,
    "delta_t" => delta_t,
    "delta_ball" => delta_ball
)

model_params = Dict{String,Any}(
    "u+" => 0.5,
    "u-" => -0.5
)

model_info = Dict{String,Any}(
    "model" => "consum",
    "dimension" => 2,
    "estimation-args" => estimation_args,
    "model-params" => model_params
)

@everywhere RET_ARRAY=COORDINATE_TYPE[0, 0]
@everywhere function consum_rhs(xy, u)
    RET_ARRAY[1] = xy[1] - xy[2]
    RET_ARRAY[2] = u
    return RET_ARRAY
end

rhss = [x->consum_rhs(x, model_params["u-"]), x->consum_rhs(x,  model_params["u+"])]
rhss_transformed =
    map(temporal_homogenization,
        map(
            f->spatial_normalization(f,bounds),
            rhss
        )
    )
step_functions = map(f->rhs2stepfct(f, delta_t), rhss_transformed)

function normalized_is_sunny(x)
    # set only the boundary dark
    lower = stepsize / 2
    upper = 1-lower
    x[1] > lower && x[2] > lower && x[1] < upper && x[2] < upper
end

mode = sp_mparallel
# mode = sp_recursive
# mode = sp_parallel


points = create_normalized_2d_grid(num_per_dim)
states = map_over_points(STATE_TYPE, normalized_is_sunny, points)
@time ret = sp_viability_kernel(
    points,
    states,
    step_functions,
    delta=delta_ball,
    computation_mode=mode,
    r=delta_t
)

points = create_normalized_2d_grid(num_per_dim)
states = map_over_points(STATE_TYPE, normalized_is_sunny, points)
@time ret = sp_viability_kernel(
    points,
    states,
    step_functions,
    delta=delta_ball,
    computation_mode=mode,
    r=delta_t
)

actual_points = points2bounds(points, bounds)

print("writing output to '$OUTFILENAME' ... ")
write_result_file(OUTFILENAME, model_info, actual_points, states)
println("done")
