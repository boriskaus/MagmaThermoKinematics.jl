"""
Diffusion2D provides 2D diffusion codes (pure 2D and axisymmetric)
"""
module Diffusion2D

# load required julia packages    
using ParallelStencil
using ParallelStencil.FiniteDifferences2D

export diffusion2D_AxiSymm_step!, diffusion2D_step!, bc2D_x!, bc2D_z!;

@init_parallel_stencil(Threads, Float64, 2);    # initialize parallel stencil in 2D



#------------------------------------------------------------------------------------------
# Solve one diffusion timestep in axisymmetric geometry, including latent heat, with spatially variable Rho, Cp and K 
@parallel function diffusion2D_AxiSymm_step!(Tnew, T, R, Rc, qr, qz, K, Kr, Kz, Rho, Cp, dt, dr, dz, L, dPhi_dt)   
    @all(Kr)    =  @av_xa(K);                                       # average K in r direction
    @all(Kz)    =  @av_ya(K);                                       # average K in z direction
    @all(qr)    =  @all(Rc).*@all(Kr).*@d_xa(T)./dr;                # heatflux in r
    @all(qz)    =            @all(Kz).*@d_ya(T)./dz;                # heatflux in z
    @inn(Tnew)  =  @inn(T) + dt./(@inn(Rho)*@inn(Cp)).* 
                                (1.0./@inn(R).*@d_xi(qr)./dr  +     # 2nd derivative in r
                                               @d_yi(qz)./dz) +     # 2nd derivative in z
                                @inn(Rho)*L*@inn(dPhi_dt);          # latent heat

    return
end

# Solve one diffusion timestep in 2D geometry, including latent heat, with spatially variable Rho, Cp and K 
@parallel function diffusion2D_step!(Tnew, T, qx, qz, K, Kx, Kz, Rho, Cp, dt, dx, dz, L, dPhi_dt)   
    @all(Kx)    =  @av_xa(K);                                       # average K in x direction
    @all(Kz)    =  @av_ya(K);                                       # average K in z direction
    @all(qx)    =  @all(Kx).*@d_xa(T)./dx;                          # heatflux in x
    @all(qz)    =  @all(Kz).*@d_ya(T)./dz;                          # heatflux in z
    @inn(Tnew)  =  @inn(T) + dt./(@inn(Rho)*@inn(Cp)).* 
                                 (@d_xi(qx)./dx + @d_yi(qz)./dz) +  # 2nd derivative 
                                 @inn(Rho)*L*@inn(dPhi_dt);         # latent heat

    return
end


# Set x- boundary conditions to be zero-flux
@parallel_indices (iy) function bc2D_x!(T::Data.Array) # apply zero flux BC's at the x-side boundaries
    T[1  , iy] = T[2    , iy]
    T[end, iy] = T[end-1, iy]
    return
end

# Set z- boundary conditions to be zero-flux
@parallel_indices (ix) function bc2D_z!(T::Data.Array) # apply zero flux BC's at side boundaries
    T[ix,1 ]    = T[ix, 2    ]
    T[ix, end]  = T[ix, end-1]
    return
end

end



"""
Diffusion3D provides 2D diffusion codes (pure 2D and axisymmetric)
"""
module Diffusion3D

# load required julia packages      
using ParallelStencil 
using ParallelStencil.FiniteDifferences3D

export diffusion3D_step!



# Solve one diffusion timestep in 3D geometry, including latent heat, with spatially variable Rho, Cp and K 
#  Note: needs the 3D stencil routines; hence part is commented
@parallel function diffusion3D_step!(Tnew, T, qx, qy, qz, K, Kx, Ky, Kz, Rho, Cp, dt, dx, dy, dz, L, dPhi_dt)   
    @all(Kx)    =  @av_xa(K);                                       # average K in x direction
    @all(Ky)    =  @av_ya(K);                                       # average K in y direction
    @all(Kz)    =  @av_za(K);                                       # average K in z direction
    @all(qx)    =  @all(Kx).*@d_xa(T)./dx;                          # heatflux in x
    @all(qy)    =  @all(Ky).*@d_ya(T)./dy;                          # heatflux in y
    @all(qz)    =  @all(Kz).*@d_za(T)./dz;                          # heatflux in z

    @inn(Tnew)  =  @inn(T) + dt./(@inn(Rho)*@inn(Cp)).* 
                 (@d_xi(qx)./dx + @d_yi(qz)./dy + @d_zi(qz)./dz) +  # 2nd derivative 
                               @inn(Rho)*L*@inn(dPhi_dt);           # latent heat

    return
end

end
