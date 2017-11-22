include("Viability.jl")

dim = 2
num_per_dim = unsigned(50)
num_total = num_per_dim ^ 2
bounds = COORDINATE_TYPE[-1 1; -1 1]

sd_boundary = 0.5

estimation_args = Dict{String,Any}(
    "num-per-dim" => num_per_dim,
    "bounds" => bounds
)

model_params = Dict{String,Any}(
    "sd-boundary" => 0.5
)

model_info = Dict{String,Any}(
    "model" => "simple-test",
    "dimension" => 2,
    "estimation-args" => estimation_args,
    "model-params" => model_params
)

function step_func(x::POINT_TYPE, a::Float64, k::Float64)
    (x-a)/k
end
f1(x::POINT_TYPE) = step_func(x, 0.0, 2.0)
f2(x::POINT_TYPE) = step_func(x, 1.0, 2.0)
step_functions = [f1, f2]

function is_sunny(x::POINT_TYPE)
    squared_norm = sum(x.^2)
    squared_norm < sd_boundary^2 && ( x[1] > -0.01 || x[1] < -0.3 )
end

##################################################
# start the actual stuff for the run
##################################################

ret = 2
points = create_2d_grid(num_per_dim, bounds)
states = map_over_points(STATE_TYPE, is_sunny, points)
for i=1:2
    points = create_2d_grid(num_per_dim, bounds)
    states = map_over_points(STATE_TYPE, is_sunny, points)
    @time ret = sp_viability_kernel(
        points,
        states,
        step_functions,
        delta=0.05
    )
end

if ret == 0
    println("computed viability kernel")
else:
    println("error value during computation $ret")
end


fname = "output.txt"
print("writing output to '$fname' ... ")
write_result_file(fname, model_info, points, states)
println("done")
