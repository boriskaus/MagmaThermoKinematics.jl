"""
Diffusion2D provides 2D diffusion codes (pure 2D and axisymmetric)
"""
module Diffusion2D
export diffusion2D_AxiSymm_step!, diffusion2D_step!, bc2D_x!, bc2D_z!, bc2D_z_bottom!,
        bc2D_z_bottom_flux!, assign!, diffusion2D_AxiSymm_residual!,
        RungaKutta1!, RungaKutta2!,RungaKutta4!, update_dϕdT_Phi!, update_Tbuffer!,
        update_relaxed_picard!, Nonlinear_Diffusion_step_2D!, Numeric_params, bc2D_T!, GridArray!

using LinearAlgebra: norm
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using Parameters

using MagmaThermoKinematics.Grid

import ..compute_meltfraction_ps!, ..compute_dϕdT_ps!, ..compute_density_ps!, ..compute_heatcapacity_ps!,
       ..compute_conductivity_ps!, ..compute_radioactive_heat_ps!, ..compute_latent_heat_ps!


__init__() = @init_parallel_stencil(Threads, Float64, 2)

"""
    GridArray!(X::AbstractArray, Z::AbstractArray, Grid::GridData)

In-place function that creates 2D arrays with `x` and `z` coordinates using the `Grid` information
"""
function GridArray!(X::AbstractArray, Z::AbstractArray, Grid::GridData)
    @parallel (1:Grid.N[1], 1:Grid.N[2]) GridArray!(X, Z, Grid.coord1D[1], Grid.coord1D[2])

    return nothing
end

@parallel_indices (i,j) function GridArray!(X::AbstractArray, Z::AbstractArray, x::AbstractVector,z::AbstractVector)
        X[i,j] = x[i]
        Z[i,j] = z[j]
    return
end

@parallel_indices (i,j) function deactivate_dϕdT_ϕ_belowDepth!(dϕdT, Phi_melt, Z, minZ)
    @inbounds if Z[i,j] < minZ
        dϕdT[i,j] = 0.0
        Phi_melt[i,j] = 0.0
    end
    return
end

@parallel function update_Tbuffer!(A::AbstractArray, B::AbstractArray, C::AbstractArray)
    @all(A) = @all(B) - @all(C)
    return
end

@parallel function update_relaxed_picard!(Tupdate::AbstractArray, Tnew::AbstractArray, T_it_old::AbstractArray, ω::Number)
    @all(Tupdate) = ω*@all(Tnew) + (1.0-ω)*@all(T_it_old)
    return
end

@parallel function assign!(A::AbstractArray, B::AbstractArray, add::Number)
    @all(A) = @all(B) + add
    return
end

@parallel function assign!(A::AbstractArray, B::AbstractArray)
    @all(A) = @all(B)
    return
end

# 1th order RK update (or Euler update)
@parallel function RungaKutta1!(Tnew, T, Residual, dt)
    @all(Tnew) =  @all(T) + dt*@all(Residual)
    return
end

# 2nd order RK update
@parallel function RungaKutta2!(Tnew, T, Residual, Residual1, dt)
    @all(Tnew) =  @all(T) + dt/2.0*(@all(Residual) + @all(Residual1))
    return
end

# 4th order RK update
@parallel function RungaKutta4!(Tnew, T, Residual, Residual1,  Residual2,  Residual3, dt)
    @all(Tnew) =  @all(T) + dt/6.0*(@all(Residual) + 2.0*@all(Residual1) + 2.0*@all(Residual2) + @all(Residual3))
    return
end


"""

Various parameters that control the nonlinear solver
"""
@with_kw struct Numeric_params
    ω::Float64                  =   0.8;            # relaxation parameter for nonlinear iterations
    max_iter::Int64             =   1500;           # max. number of nonlinear iterations
    verbose::Bool               =   false;          # print info?
    convergence::Float64        =   1e-4;           # nonlinear convergence criteria
    axisymmetric::Bool          =   false;          # Axisymmetric or 2D?
    flux_bottom_BC::Bool        =   false;          # Flux bottom BC?
    flux_bottom::Float64        =   0.0;            # flux @ bottom, in case flux_bottom_BC=true
    deactivate_La_at_depth::Bool=   false;
end


"""
    Nonlinear_Diffusion_step_2D!(Arrays, Mat_tup, Phases, Grid, dt, Num = Numeric_params() )

Performs a single, nonlinear, diffusion step during which temperature dependent properties (density, heat capacity, conductivity), are updated
"""
function Nonlinear_Diffusion_step_2D!(Arrays, Mat_tup, Phases, Grid, dt, Num = Numeric_params() )
    err, iter = 1., 1
    @parallel assign!(Arrays.T_K, Arrays.T, 273.15)
    @parallel assign!(Arrays.T_it_old, Arrays.T)
    Nx, Nz = (Grid.N...,)
    if haskey(Arrays,:index)
        args1  = (;T=Arrays.T_K, P=Arrays.P, index=Arrays.index)
    else
        args1  = (;T=Arrays.T_K, P=Arrays.P)
    end
    args2  = (;z=-Arrays.Z)
    Tupdate = similar(Arrays.Tnew)                 # relaxed picard update
    Tbuffer = similar(Arrays.T)
    while err>Num.convergence && iter<Num.max_iter

        @parallel (1:Nx, 1:Nz) compute_meltfraction_ps!(Arrays.ϕ, Mat_tup, Phases, args1)
        @parallel (1:Nx, 1:Nz) compute_dϕdT_ps!(Arrays.dϕdT, Mat_tup, Phases, args1)
        @parallel (1:Nx, 1:Nz) compute_density_ps!(Arrays.Rho, Mat_tup, Phases, args1)
        @parallel (1:Nx, 1:Nz) compute_heatcapacity_ps!(Arrays.Cp, Mat_tup, Phases, args1 )
        @parallel (1:Nx, 1:Nz) compute_conductivity_ps!(Arrays.Kc, Mat_tup, Phases, args1 )
        @parallel (1:Nx, 1:Nz) compute_radioactive_heat_ps!(Arrays.Hr, Mat_tup, Phases, args2)
        @parallel (1:Nx, 1:Nz) compute_latent_heat_ps!(Arrays.Hl, Mat_tup, Phases, args1)

        # Switch off latent heat & melting below a certain depth
        if Num.deactivate_La_at_depth==true
            @parallel (1:Nx,1:Nz) deactivate_dϕdT_ϕ_belowDepth!(Arrays.dϕdT, Arrays.ϕ, Arrays.Z, Num.deactivationDepth)
        end

        # Diffusion step:
        if Num.axisymmetric==true
            @parallel diffusion2D_AxiSymm_step!(Arrays.Tnew, Arrays.T, Arrays.R, Arrays.Rc, Arrays.qx, Arrays.qz, Arrays.Kc, Arrays.Kx, Arrays.Kz,
                                                Arrays.Rho, Arrays.Cp, Arrays.Hr, Arrays.Hl, dt, Grid.Δ[1], Grid.Δ[2], Arrays.dϕdT) # axisymmetric diffusion step
        else
            @parallel diffusion2D_step!(Arrays.Tnew, Arrays.T, Arrays.qx, Arrays.qz, Arrays.Kc, Arrays.Kx, Arrays.Kz,
                                        Arrays.Rho, Arrays.Cp, Arrays.Hr, Arrays.Hl, dt, Grid.Δ[1], Grid.Δ[2], Arrays.dϕdT) # 2D diffusion step
        end

        @parallel (1:Nz) bc2D_x!(Arrays.Tnew);                      # flux-free lateral boundary conditions
        if Num.flux_bottom_BC==true
            @parallel (1:Nx) bc2D_z_bottom_flux!(Arrays.Tnew, Arrays.Kc, Grid.Δ[2], Num.flux_bottom);     # flux-free bottom BC with specified flux (if false=isothermal)
        else
            @parallel (1:Nx) bc2D_T!(Arrays.Tnew, Arrays.T);        # isothermal BC's
        end

        # Use a relaxed Picard iteration to update T used for (nonlinear) material properties:
        @parallel update_relaxed_picard!(Tupdate, Arrays.Tnew, Arrays.T_it_old, Num.ω)

        # Update T_K (used above to compute material properties)
        @parallel assign!(args1.T, Tupdate,  273.15)   # all GeoParams routines expect T in K
        @parallel update_Tbuffer!(Tbuffer, Arrays.Tnew, Arrays.T_it_old)

        # Compute error
        err     = norm(Tbuffer)/maximum(Arrays.Tnew)
        if Num.verbose==true
            println("  Nonlinear iteration $(iter), error=$(err)")
        end

        @parallel assign!(Arrays.T_it_old, Tupdate)                   # Store Tnew of last iteration step
        iter   += 1

    end
    if iter==Num.max_iter
        println("WARNING: nonlinear iterations not converging. Final error=$(err). Reduce Δt, or the relaxation parameter Num.ω (=$(Num.ω)) [0-1]")
    end
    if Num.verbose==true
        println("  ----")
    end

    return nothing
end

#------------------------------------------------------------------------------------------
# Solve one diffusion timestep in axisymmetric geometry, including latent heat, with spatially variable Rho, Cp and K
@parallel function diffusion2D_AxiSymm_step!(Tnew, T, R, Rc, qr, qz, K, Kr, Kz, Rho, Cp, H, Hl, dt, dr, dz, dϕdT)
    @all(Kr)    =  @av_xa(K);                                       # average K in r direction
    @all(Kz)    =  @av_ya(K);                                       # average K in z direction
    @all(qr)    =  @all(Rc)*@all(Kr)*@d_xa(T)/dr;                # heatflux in r
    @all(qz)    =            @all(Kz)*@d_ya(T)/dz;                # heatflux in z
    @inn(Tnew)  =  @inn(T) + dt/(@inn(Rho)*(@inn(Cp) + @inn(Hl)*@inn(dϕdT)))*
                                 ( 1.0/@inn(R)*@d_xi(qr)/dr +     # 2nd derivative in r
                                                 @d_yi(qz)/dz +     # 2nd derivative in z
                                                 @inn(H)             # heat sources
                                );

    return
end

@parallel function diffusion2D_AxiSymm_residual!(Residual, T, R, Rc, qr, qz, K, Kr, Kz, Rho, Cp, H, Hl, dr, dz, dϕdT)
    @all(Kr)        =  @av_xa(K);                                       # average K in r direction
    @all(Kz)        =  @av_ya(K);                                       # average K in z direction
    @all(qr)        =  @all(Rc)*@all(Kr)*@d_xa(T)/dr;                # heatflux in r
    @all(qz)        =            @all(Kz)*@d_ya(T)/dz;                # heatflux in z
    @inn(Residual)  =  1.0/(@inn(Rho)*(@inn(Cp) + @inn(Hl)*@inn(dϕdT)))*
                                        ( 1.0/@inn(R)*@d_xi(qr)/dr +     # 2nd derivative in r
                                                        @d_yi(qz)/dz +     # 2nd derivative in z
                                                        @inn(H)             # heat sources
                                        );

    return
end

# Solve one diffusion timestep in 2D geometry, including latent heat, with spatially variable Rho, Cp and K
@parallel function diffusion2D_step!(Tnew, T, qx, qz, K, Kx, Kz, Rho, Cp, H, Hl, dt, dx, dz,  dϕdT)
    @all(Kx)    =  @av_xa(K);                                       # average K in x direction
    @all(Kz)    =  @av_ya(K);                                       # average K in z direction
    @all(qx)    =  @all(Kx)*@d_xa(T)/dx;                            # heatflux in x
    @all(qz)    =  @all(Kz)*@d_ya(T)/dz;                            # heatflux in z
    @inn(Tnew)  =  @inn(T) + dt/(@inn(Rho)*(@inn(Cp) + @inn(Hl)*@inn(dϕdT)))*
                                 (        @d_xi(qx)/dx +
                                          @d_yi(qz)/dz +            # 2nd derivative
                                          @inn(H)                   # heat sources
                                 );

    return
end


# Set x- boundary conditions to be zero-flux
@parallel_indices (iy) function bc2D_x!(T::AbstractArray)
    T[1  , iy] = T[2    , iy]
    T[end, iy] = T[end-1, iy]
    return
end

# Set z- boundary conditions to be zero-flux
@parallel_indices (ix) function bc2D_z!(T::AbstractArray)
    T[ix,1 ]    = T[ix, 2    ]
    T[ix, end]  = T[ix, end-1]
    return
end

# Set z- boundary conditions to be isothernmal
@parallel_indices (ix) function bc2D_T!(Tnew::AbstractArray, T::AbstractArray)
    Tnew[ix,1 ]    = T[ix, 1  ]
    Tnew[ix, end]  = T[ix, end]
    return
end

# Set z- boundary conditions @ bottom to be zero-flux
@parallel_indices (ix) function bc2D_z_bottom!(T::AbstractArray)
    T[ix,1 ]    = T[ix, 2    ]
    return
end

@parallel_indices (ix) function bc2D_z_bottom_flux!(T::AbstractArray, K::AbstractArray, dz::Number, q_z::Number)
    T[ix,1 ]    = T[ix, 2    ] + q_z*dz / K[ix, 1]

    #q_z = -K*(T[ix,2]-T[ix,1])/dz
    #dz*q_z/K_z + +T[ix,2]= T[ix,1]

    return
end


end

"""
Diffusion3D provides 3D diffusion routines
"""
module Diffusion3D

# load required julia packages
using LinearAlgebra: norm
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
using Parameters
#using CUDA

using MagmaThermoKinematics.Grid

export  diffusion3D_step_varK!, bc3D_x!, bc3D_y!, bc3D_z_bottom!, bc3D_z_bottom_flux!, assign!, GridArray!,
        Numeric_params, Nonlinear_Diffusion_step_3D!, bc3D_T!

import ..compute_meltfraction_ps_3D!, ..compute_dϕdT_ps_3D!, ..compute_density_ps_3D!, ..compute_heatcapacity_ps_3D!,
        ..compute_conductivity_ps_3D!, ..compute_radioactive_heat_ps_3D!, ..compute_latent_heat_ps_3D!

__init__() = @init_parallel_stencil(Threads, Float64, 3)

@parallel function assign!(A::AbstractArray, B::AbstractArray)
    @all(A) = @all(B)
    return
end

@parallel function assign!(A::AbstractArray, B::AbstractArray, add::Number)
    @all(A) = @all(B) + add
    return
end

"""
    GridArray!(X::AbstractArray, Y::AbstractArray,   Z::AbstractArray, Grid::GridData)

In-place function that creates 3D arrays with `x`,`y` and `z` coordinates using the `Grid` information
"""
function GridArray!(X::AbstractArray, Y::AbstractArray, Z::AbstractArray, Grid::GridData)
    @parallel (1:Grid.N[1], 1:Grid.N[2], 1:Grid.N[3]) GridArray!(X, Y, Z, Grid.coord1D[1], Grid.coord1D[2], Grid.coord1D[3])

    return
end

@parallel_indices (i,j,k) function GridArray!(X::AbstractArray, Y::AbstractArray, Z::AbstractArray, x::AbstractVector, y::AbstractVector,z::AbstractVector)
        X[i,j,k] = x[i]
        Y[i,j,k] = y[j]
        Z[i,j,k] = z[k]
    return
end

@parallel_indices (i,j,k) function deactivate_dϕdT_ϕ_belowDepth!(dϕdT, Phi_melt, Z, minZ)
    @inbounds if Z[i,j,k] < minZ
        dϕdT[i,j,k] = 0.0
        Phi_melt[i,j,k] = 0.0
    end
    return
end

@parallel function update_relaxed_picard!(Tupdate::AbstractArray, Tnew::AbstractArray, T_it_old::AbstractArray, ω::Number)
    @all(Tupdate) = ω*@all(Tnew) + (1.0-ω)*@all(T_it_old)
    return
end

@parallel function update_Tbuffer!(A::AbstractArray, B::AbstractArray, C::AbstractArray)
    @all(A) = @all(B) - @all(C)
    return
end

"""

Various parameters that control the nonlinear solver
"""
@with_kw struct Numeric_params
    ω::Float64                  =   0.8;            # relaxation parameter for nonlinear iterations
    max_iter::Int64             =   1500;           # max. number of nonlinear iterations
    verbose::Bool               =   false;          # print info?
    convergence::Float64        =   1e-4;           # nonlinear convergence criteria
    axisymmetric::Bool          =   false;          # Axisymmetric or 2D?
    flux_bottom_BC::Bool        =   false;          # Flux bottom BC?
    flux_bottom::Float64        =   0.0;            # flux @ bottom, in case flux_bottom_BC=true
    deactivate_La_at_depth::Bool=   false;
end

"""
    Nonlinear_Diffusion_step_3D!(Arrays, Mat_tup, Phases, Grid, dt, Num = Numeric_params() )

Performs a single, nonlinear, diffusion step during which temperature dependent properties (density, heat capacity, conductivity), are updated
"""
function Nonlinear_Diffusion_step_3D!(Arrays, Mat_tup, Phases, Grid, dt, Num = Numeric_params() )
    err, iter = 1., 1
    @parallel assign!(Arrays.T_K, Arrays.T, 273.15)
    @parallel assign!(Arrays.T_it_old, Arrays.T)
    Nx, Ny, Nz = (Grid.N...,)
    if haskey(Arrays,:index)
        args1  = (;T=Arrays.T_K, P=Arrays.P, index=Arrays.index)
    else
        args1  = (;T=Arrays.T_K, P=Arrays.P)
    end
    args2 = (;z=-Arrays.Z)
    Tupdate = similar(Arrays.Tnew)                 # relaxed picard update
    Tbuffer = similar(Arrays.T)
    while err>Num.convergence && iter<Num.max_iter

        @parallel (1:Nx, 1:Ny, 1:Nz) compute_meltfraction_ps_3D!(Arrays.ϕ, Mat_tup, Phases, args1)
        @parallel (1:Nx, 1:Ny, 1:Nz) compute_dϕdT_ps_3D!(Arrays.dϕdT, Mat_tup, Phases, args1)
        @parallel (1:Nx, 1:Ny, 1:Nz) compute_density_ps_3D!(Arrays.Rho, Mat_tup, Phases, args1)
        @parallel (1:Nx, 1:Ny, 1:Nz) compute_heatcapacity_ps_3D!(Arrays.Cp, Mat_tup, Phases, args1 )
        @parallel (1:Nx, 1:Ny, 1:Nz) compute_conductivity_ps_3D!(Arrays.Kc, Mat_tup, Phases, args1 )
        @parallel (1:Nx, 1:Ny, 1:Nz) compute_radioactive_heat_ps_3D!(Arrays.Hr, Mat_tup, Phases, args2)
        @parallel (1:Nx, 1:Ny, 1:Nz) compute_latent_heat_ps_3D!(Arrays.Hl, Mat_tup, Phases, args1)

        # Switch off latent heat & melting below a certain depth
        if Num.deactivate_La_at_depth==true
            @parallel (1:Nx,1:Ny,1:Nz) deactivate_dϕdT_ϕ_belowDepth!(Arrays.dϕdT, Arrays.ϕ, Arrays.Z, Num.deactivationDepth)
        end

        # Diffusion step:
        @parallel diffusion3D_step_varK!(Arrays.Tnew, Arrays.T, Arrays.qx, Arrays.qy, Arrays.qz, Arrays.Kc, Arrays.Kx, Arrays.Ky, Arrays.Kz,
                                            Arrays.Rho, Arrays.Cp, Arrays.Hr, Arrays.Hl, dt, Grid.Δ[1], Grid.Δ[2], Grid.Δ[3], Arrays.dϕdT)

        @parallel (1:Ny, 1:Nz) bc3D_x!(Arrays.Tnew);                      # flux-free lateral boundary conditions
        @parallel (1:Nx, 1:Nz) bc3D_y!(Arrays.Tnew);                      # flux-free lateral boundary conditions
        if Num.flux_bottom_BC==true
            @parallel (1:Nx,1:Ny) bc3D_z_bottom_flux!(Arrays.Tnew, Arrays.Kc, Grid.Δ[end], Num.flux_bottom);     # flux-free bottom BC with specified flux (if false=isothermal)
        else
            @parallel (1:Nx,1:Ny) bc3D_T!(Arrays.Tnew, Arrays.T);        # isothermal BC's
        end

        # Use a relaxed Picard iteration to update T used for (nonlinear) material properties:
        @parallel update_relaxed_picard!(Tupdate, Arrays.Tnew, Arrays.T_it_old, Num.ω)

        # Update T_K (used above to compute material properties)
        @parallel assign!(args1.T, Tupdate,  273.15)   # all GeoParams routines expect T in K
        @parallel update_Tbuffer!(Tbuffer, Arrays.Tnew, Arrays.T_it_old)

        # Compute error
        err     = norm(Tbuffer)/maximum(Arrays.Tnew)
        if Num.verbose==true
            println("  Nonlinear iteration $(iter), error=$(err)")
        end

        @parallel assign!(Arrays.T_it_old, Tupdate)                   # Store Tnew of last iteration step
        iter   += 1

    end
    if iter==Num.max_iter
        println("WARNING: nonlinear iterations not converging. Final error=$(err). Reduce Δt, or the relaxation parameter Num.ω (=$(Num.ω)) [0-1]")
    end
    if Num.verbose==true
        println("  ----")
    end

    return nothing
end


# Solve one diffusion timestep in 3D geometry, including latent heat, with spatially variable Rho, Cp and K
#  Note: needs the 3D stencil routines; hence part is commented
@parallel function diffusion3D_step_varK!(Tnew, T, qx, qy, qz, K, Kx, Ky, Kz, Rho, Cp, H, Hl, dt, dx, dy, dz, dϕdT)
    @all(Kx)    =  @av_xa(K);                                       # average K in x direction
    @all(Ky)    =  @av_ya(K);                                       # average K in y direction
    @all(Kz)    =  @av_za(K);                                       # average K in z direction
    @all(qx)    =  @all(Kx)*@d_xa(T)/dx;                          # heatflux in x
    @all(qy)    =  @all(Ky)*@d_ya(T)/dy;                          # heatflux in y
    @all(qz)    =  @all(Kz)*@d_za(T)/dz;                          # heatflux in z

    @inn(Tnew)  =  @inn(T) + dt/(@inn(Rho)*(@inn(Cp) + @inn(Hl)*@inn(dϕdT)))*
                   (  @d_xi(qx)/dx +
                      @d_yi(qy)/dy +
                      @d_zi(qz)/dz +               # 2nd derivative
                      @inn(H)  );                   # heat sources

    return
end

# Set x- boundary conditions to be zero-flux
@parallel_indices (iy,iz) function bc3D_x!(T)
    T[1  , iy,iz] = T[2    , iy,iz]
    T[end, iy,iz] = T[end-1, iy,iz]
    return
end

# Set y- boundary conditions to be zero-flux
@parallel_indices (ix,iz) function bc3D_y!(T)
    T[ix  , 1,iz] = T[ix, 2,iz]
    T[ix, end,iz] = T[ix, end-1,iz]
    return
end

# Set z- boundary conditions @ bottom to be zero-flux
@parallel_indices (ix,iy) function bc3D_z_bottom!(T::AbstractArray)
    T[ix,iy,1 ]    = T[ix, iy,2  ]
    return
end

@parallel_indices (ix,iy) function bc3D_z_bottom_flux!(T::AbstractArray, K::AbstractArray, dz::Number, q_z::Number)
    T[ix,iy,1 ]    = T[ix, iy, 2    ] + q_z*dz / K[ix, iy, 1]

    return
end

# Set z- boundary conditions to be isothernmal
@parallel_indices (ix, iy) function bc3D_T!(Tnew::AbstractArray, T::AbstractArray)
    Tnew[ix,iy,1 ]    = T[ix, iy, 1  ]
    Tnew[ix,iy, end]  = T[ix, iy, end]
    return
end

end
