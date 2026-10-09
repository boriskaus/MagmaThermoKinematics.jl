"""
    SolidFraction(T, Phi_o, dt)

Compute the solid fraction `Phi` (= 1 - melt fraction) from the temperature `T`
[Celsius] with the parameterized melting model, and its rate `dPhi_dt` over the
time step `dt`. Returns `(Phi, dPhi_dt)`.
"""
function SolidFraction(T::Array, Phi_o::Array, dt::Float64)
    Phi_new = zero(Phi_o)
    dPhi_dt = zero(Phi_o)
    SolidFraction_Parameterized!(T, Phi_o, Phi_new, dPhi_dt, dt)
    return Phi_new, dPhi_dt
end


function SolidFraction_Parameterized!(T::Array, Phi_o::Array, Phi::Array, dPhi_dt::Array, dt::Float64)
    # Compute the melt fraction of the domain, assuming T=Celcius
    # Taken from L.Caricchi (pers. comm.)

    # Also compute dPhi/dt, which is used to compute latent heat

    #Theta      =   (800.0 .- T)./23.0;
    #Phi        =   1.0 .- 1.0./(1.0 .+ exp.(Theta));


    Phi .= 1.0 .- 1.0 ./ (1.0 .+ exp.((800.0 .- T) ./ 23.0))

    dPhi_dt .= (Phi .- Phi_o) ./ dt
    return Phi_o .= Phi

end
