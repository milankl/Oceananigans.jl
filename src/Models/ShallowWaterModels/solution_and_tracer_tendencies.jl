using Oceananigans.Advection
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using Oceananigans.Coriolis
using Oceananigans.Operators
using Oceananigans.TurbulenceClosures: ∇_dot_qᶜ

@inline squared(i, j, k, grid, ϕ) = @inbounds ϕ[i, j, k]^2

@inline half_g_h²(i, j, k, grid, h, g) = @inbounds 1/2 * g * h[i, j, k]^2

@inline x_pressure_gradient(i, j, k, grid, g, h, formulation) = ∂xᶠᶜᶜ(i, j, k, grid, half_g_h², h, g)
@inline y_pressure_gradient(i, j, k, grid, g, h, formulation) = ∂yᶜᶠᶜ(i, j, k, grid, half_g_h², h, g)

@inline x_pressure_gradient(i, j, k, grid, g, h, ::VectorInvariantFormulation) = g * ∂xᶠᶜᶜ(i, j, k, grid, h)
@inline y_pressure_gradient(i, j, k, grid, g, h, ::VectorInvariantFormulation) = g * ∂yᶜᶠᶜ(i, j, k, grid, h)

@inline x_bathymetry(i, j, k, grid, g, h, hb, formulation) = g * h[i, j, k] * ∂xᶠᶜᶜ(i, j, k, grid, hb)
@inline y_bathymetry(i, j, k, grid, g, h, hb, formulation) = g * h[i, j, k] * ∂yᶜᶠᶜ(i, j, k, grid, hb)

@inline x_bathymetry(i, j, k, grid, g, h, hb, ::VectorInvariantFormulation) = g * ∂xᶠᶜᶜ(i, j, k, grid, hb)
@inline y_bathymetry(i, j, k, grid, g, h, hb, ::VectorInvariantFormulation) = g * ∂yᶜᶠᶜ(i, j, k, grid, hb)


"""
Compute the tendency for the x-directional transport, uh
"""
@inline function uh_solution_tendency(i, j, k, grid,
                                      gravitational_acceleration,
                                      advection,
                                      coriolis,
                                      closure,
                                      bathymetry,
                                      solution,
                                      tracers,
                                      diffusivities,
                                      forcings,
                                      clock,
                                      formulation)

    g = gravitational_acceleration

    return ( - div_hUu(i, j, k, grid, advection, solution, formulation)
             - x_pressure_gradient(i, j, k, grid, gravitational_acceleration, solution.h, formulation)
             - x_f_cross_U(i, j, k, grid, coriolis, solution)
             + x_bathymetry(i, j, k, grid, g, solution.h, bathymetry, formulation)
             + forcings[1](i, j, k, grid, clock, merge(solution, tracers)))
end

"""
Compute the tendency for the y-directional transport, vh.
"""
@inline function vh_solution_tendency(i, j, k, grid,
                                      gravitational_acceleration,
                                      advection,
                                      coriolis,
                                      closure,
                                      bathymetry,
                                      solution,
                                      tracers,
                                      diffusivities,
                                      forcings,
                                      clock,
                                      formulation)

     g = gravitational_acceleration

    return ( - div_hUv(i, j, k, grid, advection, solution, formulation)
             - y_pressure_gradient(i, j, k, grid, gravitational_acceleration, solution.h, formulation)
             - y_f_cross_U(i, j, k, grid, coriolis, solution)
             + y_bathymetry(i, j, k, grid, g, solution.h, bathymetry, formulation)
             + forcings[2](i, j, k, grid, clock, merge(solution, tracers)))
end

"""
Compute the tendency for the height, h.
"""
@inline function h_solution_tendency(i, j, k, grid,
                                     gravitational_acceleration,
                                     advection,
                                     coriolis,
                                     closure,
                                     bathymetry,
                                     solution,
                                     tracers,
                                     diffusivities,
                                     forcings,
                                     clock,
                                     formulation)

    return ( - div_Uh(i, j, k, grid, advection, solution, formulation)
             + forcings.h(i, j, k, grid, clock, merge(solution, tracers)))
end

@inline function tracer_tendency(i, j, k, grid,
                                 val_tracer_index::Val{tracer_index},
                                 advection,
                                 closure,
                                 solution,
                                 tracers,
                                 diffusivities,
                                 forcing,
                                 clock,
                                 formulation) where tracer_index

    @inbounds c = tracers[tracer_index]

    return ( - div_Uc(i, j, k, grid, advection, solution, c, formulation) 
             + c_div_U(i, j, k, grid, solution, c, formulation)         
             - ∇_dot_qᶜ(i, j, k, grid, closure, c, val_tracer_index, clock, diffusivities, tracers, nothing)
             + forcing(i, j, k, grid, clock, merge(solution, tracers)) 
            )
end
