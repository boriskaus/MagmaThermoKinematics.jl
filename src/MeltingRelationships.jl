module MeltingRelationships


export SolidFraction


"""
    SolidFraction computes the solid fraction (= (1-phi) as a function of T

"""
function SolidFraction(T::Array, Phi::Array, Phi_o::Array, dPhi_dt::Array, dt::Float64)
    # Compute the melt fraction of the domain, assuming T=Celcius
    # Taken from L.Caricchi (pers. comm.)

    # Also compute dPhi/dt, which is used to compute latent heat

    Theta      =   (800.0 .- T)./23.0;
    Phi       .=   1.0 .- 1.0./(1.0 .+ exp.(Theta)); 
    dPhi_dt   .=   (Phi .- Phi_o)./dt;  
    Phi_o     .=   Phi;

    return
end



end