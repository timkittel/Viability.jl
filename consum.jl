include("Viability.jl")


OUTFILENAME = "consum.txt"

dim = 2
num_per_dim = unsigned(101)
stepsize=1/(num_per_dim-1)
println("stepsize $stepsize")
delta_t = 6*stepsize
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

RET_ARRAY=COORDINATE_TYPE[0, 0]
function consum_rhs(xy, u)
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
    lower = stepsize / 2
    upper = 1-lower
    x[1] > lower && x[2] > lower && x[1] < upper && x[2] < upper
end

points = create_normalized_2d_grid(num_per_dim)
states = map_over_points(STATE_TYPE, normalized_is_sunny, points)
# states = ones(STATE_TYPE, num_total)

# x = COORDINATE_TYPE[0.2, 0.4]
# println("start evaluations")
# println(rhss[1](x))
# println("x $x")
# println(ndims(x), " ", size(x))
# println(rhss_transformed[1](x))


@time ret = sp_viability_kernel(
    points,
    states,
    step_functions,
    delta=delta_ball
)

actual_points = points2bounds(points, bounds)

print("writing output to '$OUTFILENAME' ... ")
write_result_file(OUTFILENAME, model_info, actual_points, states)
println("done")
