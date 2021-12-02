using Oceananigans.Solvers
using Oceananigans.Operators
using Oceananigans.Architectures
using Oceananigans.Fields: ReducedField
using Oceananigans.Solvers: MatrixIterativeSolver
import Oceananigans.Solvers: solve!

struct MatrixImplicitFreeSurfaceSolver{V, S, R}
    vertically_integrated_lateral_areas :: V
    matrix_iterative_solver :: S
    right_hand_side :: R
end

"""
    PCGImplicitFreeSurfaceSolver(arch::AbstractArchitecture, grid, settings)

Return a the framework for solving the elliptic equation with one of the iterative solvers of IterativeSolvers.jl
with a sparse matrix formulation.
    
```math
[∇ ⋅ H ∇ - Az / (g Δt²)] ηⁿ⁺¹ = (∇ʰ ⋅ Q★ - Az ηⁿ / Δt) / (g Δt) 
```

representing an implicit time discretization of the linear free surface evolution equation
for a fluid with variable depth `H`, horizontal areas `Az`, barotropic volume flux `Q★`, time
step `Δt`, gravitational acceleration `g`, and free surface at time-step `n` `ηⁿ`.
"""

function MatrixImplicitFreeSurfaceSolver(arch::AbstractArchitecture, grid, gravity, settings)
    
    # Initialize vertically integrated lateral face areas
    ∫ᶻ_Axᶠᶜᶜ = ReducedField(Face, Center, Nothing, arch, grid; dims=3)
    ∫ᶻ_Ayᶜᶠᶜ = ReducedField(Center, Face, Nothing, arch, grid; dims=3)

    vertically_integrated_lateral_areas = (xᶠᶜᶜ = ∫ᶻ_Axᶠᶜᶜ, yᶜᶠᶜ = ∫ᶻ_Ayᶜᶠᶜ)

    compute_vertically_integrated_lateral_areas!(vertically_integrated_lateral_areas, grid, arch)

    right_hand_side = arch_array(arch, zeros(eltype(grid), grid.Nx * grid.Ny)) # linearized RHS for matrix operations

    # Set maximum iterations to Nx * Ny if not set
    settings = Dict{Symbol, Any}(settings)
    maximum_iterations = get(settings, :maximum_iterations, grid.Nx * grid.Ny)
    settings[:maximum_iterations] = maximum_iterations

    coeffs = compute_coefficients(vertically_integrated_lateral_areas, grid, gravity)

    solver = MatrixIterativeSolver(coeffs;
                                   grid = grid,
                                   settings...)

    return MatrixImplicitFreeSurfaceSolver(vertically_integrated_lateral_areas, solver, right_hand_side)
end

build_implicit_step_solver(::Val{:MatrixIterativeSolver}, arch, grid, gravity, settings) =
    MatrixImplicitFreeSurfaceSolver(arch, grid, gravity, settings)

#####
##### Solve...
#####

function solve!(η, implicit_free_surface_solver::MatrixImplicitFreeSurfaceSolver, rhs, g, Δt)

    solver = implicit_free_surface_solver.matrix_iterative_solver

    solve!(η, solver, rhs, Δt)

    return nothing
end

function compute_implicit_free_surface_right_hand_side!(rhs,
                                                        implicit_solver::MatrixImplicitFreeSurfaceSolver,
                                                        g, Δt, ∫ᶻQ, η)

    solver = implicit_solver.matrix_iterative_solver
    arch = solver.architecture
    grid = solver.grid

    event = launch!(arch, grid, :xy,
                    implicit_linearized_free_surface_right_hand_side!,
                    rhs, grid, g, Δt, ∫ᶻQ, η,
		            dependencies = device_event(arch))

    return event
end

# linearized right hand side
@kernel function implicit_linearized_free_surface_right_hand_side!(rhs, grid, g, Δt, ∫ᶻQ, η)
    i, j = @index(Global, NTuple)
    Az = Azᶜᶜᵃ(i, j, 1, grid)
    δ_Q = flux_div_xyᶜᶜᵃ(i, j, 1, grid, ∫ᶻQ.u, ∫ᶻQ.v)
    t = i + grid.Nx * (j - 1)
    @inbounds rhs[t] = (δ_Q - Az * η[i, j, 1] / Δt) / (g * Δt)
end

function compute_coefficients(vertically_integrated_areas, grid, gravity)

    arch = grid.architecture

    Nx, Ny = (grid.Nx, grid.Ny)

    C  = arch_array(arch, zeros(Nx, Ny))
    Ax = arch_array(arch, zeros(Nx, Ny))
    Ay = arch_array(arch, zeros(Nx, Ny))

    ∫Ax = vertically_integrated_areas.xᶠᶜᶜ
    ∫Ay = vertically_integrated_areas.yᶜᶠᶜ

    coeff_c_size = (Nx, Ny)
    event_c = launch!(arch, grid, coeff_c_size, _compute_coefficients!, C, Ax, Ay, ∫Ax, ∫Ay, grid, gravity)
  
    wait(event_c)

    return (Ax, Ay, C)
end

@kernel function _compute_coefficients!(C, Ax, Ay, ∫Ax, ∫Ay, grid, g)
    i, j = @index(Global, NTuple)
    Ay[i, j] = ∫Ay[i, j, 1] / Δyᶜᶠᵃ(i, j, 1, grid)
    Ax[i, j] = ∫Ax[i, j, 1] / Δxᶠᶜᵃ(i, j, 1, grid)
    C[i, j] = - Azᶜᶜᵃ(i, j, 1, grid) / g
end