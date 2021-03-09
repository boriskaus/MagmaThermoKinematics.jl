# this file tests various aspects of the 
using MagmaThermoKinematics
using Plots  
using LinearAlgebra
using SpecialFunctions
using Test


const CreatePlots = false      # easy way to deactivate plotting throughout


function test_SolidFraction()
  # test solid fraction

 
  # Model parameters
  H                     =   75e3;                              # Height
    
  # Define  grid
  Nz                    =   33;                                 # resolution of coarse grid
  dz                    =   H/(Nz-1);                           # grid size [m]
  z                     =   0:dz:((Nz-1)*dz);                   # 1D coordinate arrays
  coords                =   collect(Iterators.product(z))       # generate coordinates from 1D coordinate vectors   
  Z                     =   (x->x[1]).(coords);                 # transfer coords to 3D arrays
  Grid, Spacing         =   (z), (dz);
    
    
  # Define T
  T                     =   Z./1e3.*20;
  T[ (Z.>50) .& (Z.< 60)]     .=   900;

  Phi_o                 = 0*T;
  dPhi_dt               = 0*T;
  dt                    = 1.0;
  Phi                   = 0*T;
  

  Phi, dPhi_dt = SolidFraction(T, Phi_o, dt); # call routine

  
  if CreatePlots
      p1          =   plot(T[:,1], Phi[:,1],  ylabel="Solid fraction ", xlabel="Temperature [C]",  dpi=150)
      
      plot(p1);
      png("MeltingRelationship")
  end

  out = norm(Phi[:],2)

  return out;       
end 






# ===================================================================================================

@testset "MeltingRelationShips" begin
  @test test_SolidFraction() ≈  4.132199893765409  atol=1e-8;
end;
