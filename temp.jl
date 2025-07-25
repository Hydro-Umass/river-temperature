using ModelingToolkit, MethodOfLines, DomainSets, DifferentialEquations
using Plots

@parameters t x
@variables T(..)

Dt = Differential(t)
Dx = Differential(x)
Dxx = Differential(x)^2

# physical parameters
U = 0.5       # mean velocity [m/s] (dummy value)
D = 0.2       # diffusion coefficient [m²/s] (dummy value)
H = 800.0     # heat flux [W/m²] (dummy value)
ρ = 1000.0    # water density [kg/m³]
c = 4186.0    # specific heat capacity [J/kg·K]
d = 2.5       # mean depth [m] (dummy value)
S = H/(ρ*c*d) # source term [K/s]

# domain parameters
L = 1000.0    # river length [m]
t_max = 3600.0 # simulation time [s] (1 hour)

# define the PDE
eq = Dt(T(t,x)) ~ -U*Dx(T(t,x)) + D*Dxx(T(t,x)) + S

# initial condition - Gaussian distribution centered in the river
T0(x) = 15.0 + 10.0*exp(-(x - L/2)^2/(L/4)^2)  # 15°C baseline + 10°C peak

# boundary conditions
bcs = [
    T(0,x) ~ T0(x),
    T(t,0) ~ 15.0,
    T(t,L) ~ 15.0
]

# define domains
domains = [
    t ∈ Interval(0.0, t_max),
    x ∈ Interval(0.0, L)
]

# create the PDESystem
@named pdesys = PDESystem([eq], bcs, domains, [t,x], [T(t,x)])

# discretization parameters
Nx = 100       # Number of spatial points
Δx = L/(Nx-1)  # Spatial step size

# create discretization system
discretization = MOLFiniteDifference(
    [x => Δx],  # spatial discretization
    t,          # time variable
    approx_order=4  # higher order approximation
)

# convert the PDE system to an ODE system
prob = discretize(pdesys, discretization)

# solve the problem
sol = solve(prob, Tsit5(), saveat=300.0)  # save every 5 minutes

# post-processing and visualization
x_grid = 0:Δx:L  # spatial grid points

# create animation
anim = @animate for i in 1:length(sol.t)
    plot(x_grid, sol[T(t,x)][i,:],
         xlabel="Distance along river (m)",
         ylabel="Temperature (°C)",
         title="River Temperature at t = $(sol.t[i]) seconds",
         ylim=(10, 26),
         legend=false,
         linewidth=2)
end

# save animation (uncomment to save)
gif(anim, "river_temp.gif", fps=5)

# display final temperature profile
plot(x_grid, sol[T(t,x)][end,:],
     xlabel="Distance along river (m)",
     ylabel="Temperature (°C)",
     title="Final Temperature Profile after 1 hour",
     legend=false,
     linewidth=2)
