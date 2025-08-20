using CSV, DataFrames
using Dates
using Interpolations
using DifferentialEquations
using Lux#, LuxCUDA
using SciMLSensitivity, Zygote, ComponentArrays
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using Random, Statistics, CurveFit
using Plots

const ρ = Float32(1000.0) # Water density [kg/m³]
const cₚ = Float32(41860.0) # Specific heat [J/kg/K]
const dt = 24 # time steps per day

function load_site_data(site::Int, start_date::Date, end_date::Date)
    # temperature data
    temp = CSV.File("data/selected_reaches_temp_usgs.csv", missingstring="NA") |> DataFrame
    tdata0 = filter(:site_id => s -> occursin("$(site)", s), temp)
    # streamflow data
    q = CSV.File("data/selected_reaches_streamflow_usgs.csv", missingstring="NA") |> DataFrame
    qdata0 = filter(:site_id => s -> occursin("$(site)", s), q)
    # meteorological data
    met = CSV.File("data/selected_reaches_era5.csv", missingstring="NA") |> DataFrame
    rename!(met, :valid_time => :date)
    mdata0 = filter(:site_no => s -> s == site, met)
    # filter for dates
    tdata = filter(:date => t -> start_date <= t <= end_date, tdata0)
    qdata = filter(:date => t -> start_date <= t <= end_date, qdata0)
    mdata = filter(:date => t -> start_date <= t <= end_date, mdata0)
    return tdata, qdata, mdata, tdata0, qdata0, mdata0
end

function build_interpolations(mdata::DataFrame, qdata::DataFrame, width::Float64, dist::Float64)
    date_range = minimum(mdata[:, :date]):Day(1):maximum(mdata[:, :date])
    time_points = Float32.(collect(0.0:1.0:(length(date_range) - 1)) .* dt)
    H_vals = Float32.(sum.(eachrow(mdata[: , [:slhf, :sshf, :ssr, :str]])))
    # effective depth from discharge
    d_vals = Float32.(0.1 .* qdata[:, :discharge_cms].^0.4)
    # river volume
    V_vals = Float32.(d_vals .* width .* dist)
    Q_vals = Float32.(qdata[:, :discharge_cms])
    Tair_vals = Float32.(mdata[:, :t2m])
    Hi = LinearInterpolation(time_points, H_vals)
    di = LinearInterpolation(time_points, d_vals)
    Vi = LinearInterpolation(time_points, V_vals)
    Qi = LinearInterpolation(time_points, Q_vals)
    Tai = LinearInterpolation(time_points, Tair_vals)
    return Hi, di, Vi, Qi, Tai
end

function bc_temp_from_air_temp(mdata::DataFrame, mdata0::DataFrame, tdata0::DataFrame)
    df = outerjoin(mdata0, tdata0, on=:date, makeunique=true) |> dropmissing
    f  = fit(Polynomial, df[:, :t2m], df[:, :mean_temp_c], 2)
    date_range  = minimum(mdata[:, :date]):Day(1):maximum(mdata[:, :date])
    time_points = collect(0.0:1.0:(length(date_range) - 1)) .* dt
    T_itp = LinearInterpolation(time_points, f.(mdata[:, :t2m]))
    return T_itp
end

function lumped_temp!(dT, T, p, t)
    Hi, di, Qi, Vi, Ti = p
    H = Hi(t)
    d  = di(t)
    Q  = Qi(t)
    V  = Vi(t)
    Tv = Ti(t)
    dT[1] = H / (ρ * cₚ * d) + (Q * Tv - Q * T[1]) / V
end

# example main function to show solving  an ODE with upstream boundary condition estimated from regression to air temperature
function main_temp_regression()
    sites = [1480870, 1481000, 1481500]
    site = sites[3]
    start_date =  Date(2010, 10, 1)
    end_date = Date(2011, 9, 30)
    tdata, qdata, mdata, tdata0, _, mdata0 = load_site_data(site, start_date, end_date)
    # assume rectangular section
    riv_width = 20.0
    riv_length = 1000.0
    Hi, di, Vi, Qi, _ = build_interpolations(mdata, qdata, riv_width, riv_length)
    date_range = start_date:Day(1):end_date
    time_points = collect(0.0:length(date_range)-1) * dt
    Ti = bc_temp_from_air_temp(mdata, mdata0, tdata0)
    T0 = [tdata[1, :mean_temp_c]]
    tspan = (time_points[1], time_points[end])
    prob = ODEProblem(lumped_temp!, T0 ,tspan, (Hi, di, Qi, Vi, Ti))
    sol = solve(prob, Tsit5(), saveat=time_points)
    plot(date_range, [u[1] for u in sol.u], label="Simulation")
    scatter!(date_range[Dates.value.(tdata[:, :date] - tdata[1, :date]) .+ 1], tdata[:, :mean_temp_c], label="Observations")
    ylabel!("Temperature (℃)")
end

function lumped_temp_nn!(dT, T, p, t)
    H = Hi(t)
    d  = di(t)
    Q  = Qi(t)
    V  = Vi(t)
    Tair = Tai(t)
    Tv = dudt([(Tair - Tamean)/Tastd, (H - Hmean)/Hstd], p, st)[1][1]
    dT[1] = H / (ρ * cₚ * d) + (Q * Tv - Q * T[1]) / V
end

function main_temp_nn()
    sites = [1480870, 1481000, 1481500]
    site = sites[3]
    start_date =  Date(2010, 10, 1)
    end_date = Date(2011, 9, 30)
    tdata, qdata, mdata, tdata0, _, mdata0 = load_site_data(site, start_date, end_date)
    # assume rectangular section
    riv_width = 20.0
    riv_length = 1000.0
    date_range = start_date:Day(1):end_date
    time_points = Float32.(collect(0.0:length(date_range)-1) * dt)
    Hi, di, Vi, Qi, Tai = build_interpolations(mdata, qdata, riv_width, riv_length)
    Hmean, Hstd = mean(Hi), std(Hi)
    Tamean, Tastd = mean(Tai), std(Tai)
    # build neural network model
    rng = Xoshiro(0)
    dudt = Lux.Chain(Dense(2, 20, tanh), Dense(20, 1))
    p, st = Lux.setup(rng, dudt)
    p = ComponentArray(p)
    T0 = Float32.([tdata[1, :mean_temp_c]])
    tspan = (time_points[1], time_points[end])
    prob = ODEProblem(lumped_temp_nn!, T0, tspan)
    function predict_nn(p)
        sol = solve(prob, Tsit5(), saveat=time_points, p=p)
        Array(sol)
    end
    function loss_nn(p)
        pred = predict_nn(p)
        i = Dates.value.(tdata[:, :date] - tdata[1, :date]) .+ 1
        sum(abs2, tdata[:, :mean_temp_c] .- pred[1, i])
    end
    iter = 0
    function callback(state, l)
        global iter += 1
        iter % 5 == 0 &&
            @info "Epoch: $(iter) - Loss: $(l)"
        return false
    end
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss_nn(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, p)
    res = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.1); callback = callback, maxiters = 500)
    optprob2 = remake(optprob; u0 = res.u)
    res2 = Optimization.solve(optprob2, Optim.BFGS(; initial_stepnorm = 0.01); callback, allow_f_increases = false)
    pred = predict_nn(res2.u)[1, :]
    plot(date_range, pred, label="Simulation")
    scatter!(date_range[Dates.value.(tdata[:, :date] - tdata[1, :date]) .+ 1], tdata[:, :mean_temp_c], label="Observations")
    ylabel!("Temperature (℃)")
end
