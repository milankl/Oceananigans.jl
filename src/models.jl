using .TurbulenceClosures
using .TurbulenceClosures: ν₀, κ₀

mutable struct Model{TS, E, A<:AbstractArchitecture, G, T, B, R, U, C, Φ, F, 
                     BCS, S, K, OW, DI, Θ} <: AbstractModel

           architecture :: A         # Computer `Architecture` on which `Model` is run
                   grid :: G         # Grid of physical points on which `Model` is solved
                  clock :: Clock{T}  # Tracks iteration number and simulation time of `Model`
               buoyancy :: B         # Set of parameters for buoyancy model
               coriolis :: R         # Set of parameters for the background rotation rate of `Model`
             velocities :: U         # Container for velocity fields `u`, `v`, and `w`
                tracers :: C         # Container for tracer fields
              pressures :: Φ         # Container for hydrostatic and nonhydrostatic pressure
                forcing :: F         # Container for forcing functions defined by the user
                closure :: E         # Diffusive 'turbulence closure' for all model fields
    boundary_conditions :: BCS       # Container for 3d bcs on all fields
            timestepper :: TS        # Object containing timestepper fields and parameters
         poisson_solver :: S         # Poisson Solver
          diffusivities :: K         # Container for turbulent diffusivities
         output_writers :: OW        # Objects that write data to disk
            diagnostics :: DI        # Objects that calc diagnostics on-line during simulation
             parameters :: Θ         # Container for arbitrary user-defined parameters
end

"""
    Model(; grid, kwargs...)

Construct an `Oceananigans.jl` model on `grid`.

Important keyword arguments include

    - `grid`: (required) The resolution and discrete geometry on which `model` is solved.
              Currently the only option is `RegularCartesianGrid`.

    - `architecture`: `CPU()` or `GPU()`. The computer architecture used to time-step `model`.

    - `float_type`: `Float32` or `Float64`. The floating point type used for `model` data.

    - `closure`: The turbulence closure for `model`. See `TurbulenceClosures`.

    - `buoyancy`: Buoyancy model parameters.

    - `coriolis`: Parameters for the background rotation rate of the model.

    - `forcing`: User-defined forcing functions that contribute to solution tendencies.

    - `boundary_conditions`: User-defined boundary conditions for model fields. Can be either
                             `SolutionBoundaryConditions` or `ModelBoundaryConditions`.
                             See `BoundaryConditions`, `HorizontallyPeriodicSolutionBCs` and `ChannelSolutionBCs`.

    - `parameters`: User-defined parameters for use in user-defined forcing functions and boundary
                    condition functions.
"""
function Model(;
                   grid, # model resolution and domain
           architecture = CPU(), # model architecture
             float_type = Float64,
                tracers = (:T, :S),
                closure = ConstantIsotropicDiffusivity(float_type, ν=ν₀, κ=κ₀), # diffusivity / turbulence closure
                  clock = Clock{float_type}(0, 0), # clock for tracking iteration number and time-stepping
               buoyancy = SeawaterBuoyancy(float_type),
               coriolis = nothing,
                forcing = Forcing(),
    boundary_conditions = HorizontallyPeriodicSolutionBCs(),
         output_writers = OrderedDict{Symbol, AbstractOutputWriter}(),
            diagnostics = OrderedDict{Symbol, AbstractDiagnostic}(),
             parameters = nothing, # user-defined container for parameters in forcing and boundary conditions
    # Velocity fields, tracer fields, pressure fields, and time-stepper initialization
             velocities = VelocityFields(architecture, grid),
              pressures = PressureFields(architecture, grid),
          diffusivities = TurbulentDiffusivities(architecture, grid, tracers(tracers), closure),
            timestepper = AdamsBashforthTimestepper(float_type, architecture, grid, tracernames(tracers), 0.125),
    # Solver for Poisson's equation
         poisson_solver = PoissonSolver(architecture, PoissonBCs(boundary_conditions), grid)
    )

    if architecture == GPU()
        !has_cuda() && throw(ArgumentError("Cannot create a GPU model. No CUDA-enabled GPU was detected!"))
        if mod(grid.Nx, 16) != 0 || mod(grid.Ny, 16) != 0
            throw(ArgumentError("For GPU models, Nx and Ny must be multiples of 16."))
        end
    end

    tracers = TracerFields(architecture, grid, tracers)

    # Regularize forcing, boundary conditions, and closure for given tracer fields
    forcing = ModelForcing(tracernames(tracers), forcing)
    boundary_conditions = ModelBoundaryConditions(tracernames(tracers), boundary_conditions)
    closure = with_tracers(tracernames(tracers), closure)

    return Model(architecture, grid, clock, buoyancy, coriolis, velocities, tracers,
                 pressures, forcing, closure, boundary_conditions, timestepper,
                 poisson_solver, diffusivities, output_writers, diagnostics, parameters)
end

"""
    ChannelModel(; kwargs...)

Construct a `Model` with walls in the y-direction. This is done by imposing
`FreeSlip` boundary conditions in the y-direction instead of `Periodic`.

kwargs are passed to the regular `Model` constructor.
"""
ChannelModel(; boundary_conditions=ChannelSolutionBCs(), kwargs...) =
    Model(; boundary_conditions=boundary_conditions, kwargs...)

function BasicChannelModel(; N, L, ν=ν₀, κ=κ₀, float_type=Float64,
                           boundary_conditions=ChannelSolutionBCs(), kwargs...)

    grid = RegularCartesianGrid(float_type, N, L)
    closure = ConstantIsotropicDiffusivity(float_type, ν=ν, κ=κ)

    return Model(; float_type=float_type, grid=grid, closure=closure,
                 boundary_conditions=boundary_conditions, kwargs...)
end

"""
    BasicModel(; N, L, ν=ν₀, κ=κ₀, float_type=Float64, kwargs...)

Construct a "Basic" `Model` with resolution `N`, domain extent `L`,
precision `float_type`, and constant isotropic viscosity and diffusivity `ν`, and `κ`.

Additional `kwargs` are passed to the regular `Model` constructor.
"""
function BasicModel(; N, L, ν=ν₀, κ=κ₀, float_type=Float64, kwargs...)
    grid = RegularCartesianGrid(float_type, N, L)
    closure = ConstantIsotropicDiffusivity(float_type, ν=ν, κ=κ)
    return Model(; float_type=float_type, grid=grid, closure=closure, kwargs...)
end

"""
    NonDimensionalModel(; N, L, Re, Pr=0.7, Ri=1, Ro=Inf, float_type=Float64, kwargs...)

Construct a "Non-dimensional" `Model` with resolution `N`, domain extent `L`,
precision `float_type`, and the four non-dimensional numbers:

    * `Re = U λ / ν` (Reynolds number)
    * `Pr = U λ / κ` (Prandtl number)
    * `Ri = B λ U²`  (Richardson number)
    * `Ro = U / f λ` (Rossby number)

for characteristic velocity scale `U`, length-scale `λ`, viscosity `ν`,
tracer diffusivity `κ`, buoyancy scale (or differential) `B`, and
Coriolis parameter `f`.

Note that `N`, `L`, and `Re` are required.

Additional `kwargs` are passed to the regular `Model` constructor.
"""
function NonDimensionalModel(; N, L, Re, Pr=0.7, Ri=1, Ro=Inf, float_type=Float64, kwargs...)

         grid = RegularCartesianGrid(float_type, N, L)
      closure = ConstantIsotropicDiffusivity(float_type, ν=1/Re, κ=1/(Pr*Re))
     coriolis = VerticalRotationAxis(float_type, f=1/Ro)

     buoyancy = SeawaterBuoyancy(float_type, 
                    gravitational_acceleration = Ri, 
                    equation_of_state = LinearEquationOfState(float_type, α=1, β=0)
                )

    return Model(; float_type=float_type, grid=grid, closure=closure,
                   coriolis=coriolis, buoyancy=buoyancy, kwargs...)
end
