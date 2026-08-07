# various routines that are shared between the 2D and 3D MTK_GMG routines
"""
    MTK_GMG
This contains various user callback routines that are shared between the 2D and 3D MTK_GMG routines.
You can overwrite this in your own code to customize the simulation.

"""
module MTK_GMG

using Parameters
using GeoParams
using InjectSills
using GeophysicalModelGenerator
using StructArrays
using MagmaThermoKinematics.Grid
import MagmaThermoKinematics: NumericalParameters, SillParameters, TimeDependentProperties
import MagmaThermoKinematics: update_Tvec!, inject_sills, km³, kyr, Myr
import MagmaThermoKinematics: PhasesFromTracers!
import MagmaThermoKinematics: erupt_magma!, EruptionParameters, magma_density_fn
import MagmaThermoKinematics: FreeSurfaceParameters, init_free_surface, apply_free_surface!, advect_surface!, advect_phases!
import MagmaThermoKinematics: stamp_phase_inside_sill!
SecYear = 3600*24*365.25;

@inline _root_module() = parentmodule(@__MODULE__)
@inline DataArray(x) = getproperty(getproperty(_root_module(), :Data), :Array)(x)

@inline function CreateArrays2D(args...)
    fields2d = getproperty(_root_module(), :Fields2D)
    f = getproperty(fields2d, :CreateArrays)
    return f(args...)
end

@inline function CreateArrays3D(args...)
    fields3d = getproperty(_root_module(), :Fields3D)
    f = getproperty(fields3d, :CreateArrays)
    return f(args...)
end

@inline _active_sill(Dikes) = isnothing(Dikes.sill) ? error("SillParameters requires a valid `sill` object") : Dikes.sill
@inline _sill_radius_m(sill) = sill.W.val/2   # InjectSills stores W as the full width (diameter) ⇒ radius = W/2

#using CUDA

"""
    Analytical geotherm used for the UCLA setups, which includes radioactive heating
"""
function AnalyticalGeotherm!(T, Z, Tsurf, qm, qs, k, hr)
    T      .=  @. Tsurf - (qm/k)*Z + (qs-qm)*hr/k*( 1.0 - exp(Z/hr))
    return nothing
end

"""
    Tracers = MTK_inject_dikes(Grid, Num, Arrays, Mat_tup, Dikes, Tracers, Tnew_cpu)

Function that injects dikes once in a while
"""
function MTK_inject_dikes(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters, Tracers::StructVector, Tnew_cpu)

    if floor(Num.time/Dikes.InjectionInterval)> Dikes.sill_inj
        Dikes.sill_inj = floor(Num.time/Dikes.InjectionInterval)                 # Keeps track on what was injected already
        if Num.dim==2
            T_bottom  =   Array(@view Arrays.T[:,1])
        else
            T_bottom  =   Array(@view Arrays.T[:,:,1])
        end
        sill = _active_sill(Dikes)
        copyto!(Tnew_cpu, Arrays.T)

        Tracers, Tnew_cpu, Vol, poly_out, Velocity = inject_sills(Tracers, Tnew_cpu, Grid.coord1D, sill, Float64(Dikes.T_in_Celsius), Dikes.SillPhase, Dikes.nTr_dike, dike_poly=Dikes.sill_poly);     # Add dike, move hostrocks
        Dikes.sill_poly = poly_out

        if Num.flux_bottom_BC==false
            # Keep bottom T constant (advection modifies this)
            if Num.dim==2
                Tnew_cpu[:,1]     .=  T_bottom
            else
                Tnew_cpu[:,:,1]   .=  T_bottom
            end
        end

        Arrays.T           .=   DataArray(Tnew_cpu)
        Dikes.InjectVol    +=   Vol                                                     # Keep track of injected volume
        Qrate               =   Dikes.InjectVol/Num.time
        Dikes.Qrate_km3_yr  =   Qrate*SecYear/km³
        radius_m            =   _sill_radius_m(sill)
        Qrate_km3_yr_km2    =   Dikes.Qrate_km3_yr/(pi*(radius_m/1e3)^2)
        println("  Added new dike; time=$(Num.time/kyr) kyrs, total injected magma volume = $(Dikes.InjectVol/km³) km³; rate Q= $(Dikes.Qrate_km3_yr) km³yr⁻¹")

        if Num.advect_polygon==true && isempty(Dikes.sill_poly)
            Dikes.sill_poly = InjectSills.dike_polygon(sill)            # create sill polygon for the first time
        end

        if length(Mat_tup)>1
           # `Array(Arrays.Phases)` is a CPU copy, so we must capture it, mutate
           # it, and write it back — otherwise the injected SillPhase is lost.
           Phases = Array(Arrays.Phases)                    # move to CPU (copy)

           if Num.deform_hostrock
               # Deformable host rock (used with the free surface): advect the
               # whole material column (host rock + previously-injected sills) by
               # the injection displacement so it moves *with* the inflating
               # surface, then stamp the freshly-opened sill densely. No sparse-
               # tracer rebuild ⇒ no "spotty" host-rock nodes inside the sill.
               advect_phases!(Phases, Velocity, Grid)
               stamp_phase_inside_sill!(Phases, Grid, sill, Dikes.SillPhase)
           else
               # Pinned host rock (default): reconstruct phases from the (just-
               # injected) tracers and keep the initial host-rock phases.
               PhasesFromTracers!(Phases, Grid, Tracers, BackgroundPhase=Dikes.BackgroundPhase, InterpolationMethod="Constant");

               if Num.keep_init_RockPhases==true
                    Phases_init = Array(Arrays.Phases_init)
                    for i in eachindex(Phases)
                        if Phases[i] != Dikes.SillPhase
                            Phases[i] = Phases_init[i]
                        end
                    end
               end
           end
           Arrays.Phases .= DataArray(Phases)               # move back (always)
        end

    end

    return Tracers
end

"""
    fired = MTK_erupt!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructVector, Erupt::EruptionParameters, FS::FreeSurfaceParameters, Mat_tup::Tuple, Dikes::SillParameters)

Optional eruption callback, invoked once per timestep from the 2D/3D time loop.

When `Erupt.erupt` is `true` it evaluates the eruption trigger and, once it
fires, removes melt thermally — and optionally deflates the chamber — via
[`erupt_magma!`](@ref), writing the mutated `T`, `ϕ` and `dϕdT` fields back into
`Arrays`. It is a no-op (returns `false`) when eruptions are disabled. Returns
`true` when an eruption occurred. Overwrite this in your own code to customize
eruption behaviour.

When `Erupt.overpressure` is also `true`, this callback additionally builds the
`ρ(T_K, ϕ, P)` density callback ([`magma_density_fn`](@ref)) from
`Mat_tup[Erupt.magma_phase]`'s `Density` and `Solubility` laws, and the
recharge mass rate `Ṁ_in` from the change in `Dikes.InjectVol` since the last
step (× `Erupt.ρ_melt`, guarded by `Erupt._InjectVol_prev` against
double-counting), then passes them through to `erupt_magma!` along with
`Num.dt`.
"""
function MTK_erupt!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructVector, Erupt::EruptionParameters, FS::FreeSurfaceParameters, Mat_tup::Tuple, Dikes::SillParameters)
    Erupt.erupt || return false

    # when a moving free surface is active, let the eruption deflation subside it
    z_surf = FS.free_surface ? FS.z_surf : nothing

    Δt   = Num.dt
    ρ_fn = nothing
    Ṁ_in = 0.0
    if Erupt.overpressure
        Erupt.magma_phase < 0 && error("MTK_erupt!: Erupt.overpressure=true requires Erupt.magma_phase to be set to a valid Mat_tup phase index")
        idx = findfirst(mp -> mp.Phase == Erupt.magma_phase, Mat_tup)
        idx === nothing && error("MTK_erupt!: no Mat_tup entry has Phase == Erupt.magma_phase = $(Erupt.magma_phase)")
        density = Mat_tup[idx].Density[1]
        solub   = isempty(Mat_tup[idx].Solubility) ? nothing : Mat_tup[idx].Solubility[1]
        ρ_fn    = magma_density_fn(density, solub, Erupt)
        Ṁ_in    = (Dikes.InjectVol - Erupt._InjectVol_prev)/Δt * Erupt.ρ_melt
        Erupt._InjectVol_prev = Dikes.InjectVol
    end

    # `erupt_magma!` works on CPU `Array`s. On the CPU backend `Arrays.*` already
    # ARE plain CPU arrays, so we hand them straight to `erupt_magma!` and let it
    # mutate them in place — no host/device round-trip. `Array(x)` on a CPU array
    # is NOT free: it allocates a full copy, which (run every timestep, for T, ϕ,
    # dϕdT and Phases) was the dominant per-step cost. The copy-in/out is only
    # needed on the GPU, where it is the actual device→host→device transfer.
    if Num.USE_GPU
        T_cpu    = Array(Arrays.T)
        ϕ_cpu    = Array(Arrays.ϕ)
        dϕdT_cpu = Array(Arrays.dϕdT)
        Ph_cpu   = Num.deform_hostrock ? Array(Arrays.Phases) : nothing
        fired    = erupt_magma!(T_cpu, ϕ_cpu, dϕdT_cpu, Tracers, Grid, Erupt, Num.time;
                                 z_surf=z_surf, Phases=Ph_cpu, Δt=Δt, ρ=ρ_fn, Ṁ_in=Ṁ_in)
        if fired
            Arrays.T    .= DataArray(T_cpu)
            Arrays.ϕ    .= DataArray(ϕ_cpu)
            Arrays.dϕdT .= DataArray(dϕdT_cpu)
            Num.deform_hostrock && (Arrays.Phases .= DataArray(Ph_cpu))
        end
        return fired
    else
        Ph = Num.deform_hostrock ? Arrays.Phases : nothing
        return erupt_magma!(Arrays.T, Arrays.ϕ, Arrays.dϕdT, Tracers, Grid, Erupt, Num.time;
                             z_surf=z_surf, Phases=Ph, Δt=Δt, ρ=ρ_fn, Ṁ_in=Ṁ_in)
    end
end

# vertical host-rock displacement field of `sill` on the full grid (used to
# inflate the free surface at injection); non-finite core entries are zeroed
# just like in `inject_sills`.
function _injection_Dz(Grid::GridData, sill)
    coord = Grid.coord1D
    dim   = length(coord)
    if dim == 2
        coords = collect(Iterators.product(coord[1], coord[2]))
        X = (c->c[1]).(coords); Z = (c->c[2]).(coords)
        _, Dz = InjectSills.hostrock_displacement(sill, Float64.(X), Float64.(Z))
    else
        coords = collect(Iterators.product(coord[1], coord[2], coord[3]))
        X = (c->c[1]).(coords); Y = (c->c[2]).(coords); Z = (c->c[3]).(coords)
        _, _, Dz = InjectSills.hostrock_displacement(sill, Float64.(X), Float64.(Y), Float64.(Z))
    end
    @inbounds for i in eachindex(Dz)
        isfinite(Dz[i]) || (Dz[i] = 0.0)
    end
    return Dz
end

"""
    moved = MTK_free_surface!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Dikes::SillParameters, FS::FreeSurfaceParameters)

Optional free-surface callback, invoked once per timestep from the 2D/3D time
loop. When `FS.free_surface` is `true` it (1) inflates the surface by the
host-rock displacement of the most recently injected sill, and (2) stamps "air"
([`apply_free_surface!`](@ref)) onto every cell above the topography `FS.z_surf`.
Eruption deflation lowers the surface separately, inside [`MTK_erupt!`](@ref). A
no-op (returns `false`) when the free surface is disabled. Overwrite in your own
code to customize the free-surface behaviour.
"""
function MTK_free_surface!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Dikes::SillParameters, FS::FreeSurfaceParameters)
    FS.free_surface || return false
    isnothing(FS.z_surf) && (FS.z_surf = init_free_surface(Grid; z0=FS.z0, topography=FS.topography))
    z_surf = FS.z_surf

    # (1) injection inflation: raise the surface once per new sill injection, by
    # exactly the injected volume. The InjectSills displacement field is
    # full-space (symmetric about the sill plane), so only part of the injected
    # volume expresses at the surface; `conserve_volume` rescales it so the
    # surface rises by the whole injected amount (free-surface correction). The
    # target is the sill's 3D volume in 3D, and its per-unit-depth cross-section
    # (`area`) in 2D, matching the dimension of the surface integral.
    if hasproperty(Dikes, :sill) && !isnothing(Dikes.sill) && Dikes.sill_inj > FS._last_inj
        FS._last_inj = Dikes.sill_inj
        injected = length(Grid.coord1D) == 2 ? ustrip(InjectSills.area(Dikes.sill)) :
                                               ustrip(InjectSills.volume(Dikes.sill))
        advect_surface!(z_surf, _injection_Dz(Grid, Dikes.sill), Grid; conserve_volume = injected)
    end

    # (2) stamp air above the (possibly updated) topography — and back-fill rock
    # below it so the air/rock interface in the phase array tracks the surface
    # exactly (the continuous z_surf advances faster than the rounded, per-step
    # phase advection; see apply_free_surface!). The host/background phase is the
    # fallback fill where a column has no rock below the cell.
    fill_phase = hasproperty(Dikes, :BackgroundPhase) ? Dikes.BackgroundPhase : nothing
    if Num.USE_GPU
        # GPU: round-trip host↔device (the copies are the real transfer).
        T_cpu  = Array(Arrays.T)
        ϕ_cpu  = Array(Arrays.ϕ)
        Ph_cpu = Array(Arrays.Phases)
        apply_free_surface!(T_cpu, ϕ_cpu, Ph_cpu, z_surf, Grid, FS.Tair, FS.air_phase; fill_phase=fill_phase)
        Arrays.T      .= DataArray(T_cpu)
        Arrays.ϕ      .= DataArray(ϕ_cpu)
        Arrays.Phases .= DataArray(Ph_cpu)
    else
        # CPU: stamp air directly on the live arrays — no per-step copies (this
        # runs every timestep; `Array(x)` here would allocate three full-grid
        # copies in + three out for nothing).
        apply_free_surface!(Arrays.T, Arrays.ϕ, Arrays.Phases, z_surf, Grid, FS.Tair, FS.air_phase; fill_phase=fill_phase)
    end
    return true
end

"""
    MTK_display_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters)

Function that creates plots
"""
function MTK_visualize_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters)

    return nothing
end

"""
    MTK_print_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters)

Function that prints output to the REPL
"""
function MTK_print_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters)

    return nothing
end

"""
    MTK_update_TimeDepProps!(time_props::TimeDependentProperties, Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters)

Update time-dependent properties during a simulation
"""
function MTK_update_TimeDepProps!(time_props::TimeDependentProperties, Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters)
    push!(time_props.Time_vec,      Num.time);   # time
    push!(time_props.MeltFraction,  sum( Arrays.ϕ)/(Num.Nx*Num.Nz));    # melt fraction

    n_hot = sum(Arrays.T .> 700)
    if n_hot > 0
        Tav_magma_Time = mapreduce(t -> t > 700 ? t : zero(t), +, Arrays.T) / n_hot     # average T of part with magma
    else
        Tav_magma_Time = NaN;
    end
    push!(time_props.Tav_magma, Tav_magma_Time);       # average magma T
    push!(time_props.Tmax,      maximum(Arrays.T));   # maximum magma T

    return nothing
end

"""
    MTK_initialize!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructArray, Dikes::SillParameters)

Initialize temperature and phases
"""
function MTK_initialize!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructArray, Dikes::SillParameters)
    # Initalize T
    Arrays.T_init      .=   @. Num.Tsurface_Celcius - Arrays.Z*Num.Geotherm;                # Initial (linear) temperature profile

    # Open pvd file if requested
    if Num.Output_VTK
        name =  joinpath(Num.SimName,Num.SimName*".pvd")
        Num.pvd = movie_paraview(name=name, Initialize=true);
    end

    return nothing
end

"""
    Ararys = MTK_initialize_arrays(Num::NumericalParameters)

Initialize arrays used in the computations
"""
function MTK_initialize_arrays(Num::NumericalParameters)

    if Num.dim==2
        Arrays = CreateArrays2D(Dict( (Num.Nx,  Num.Nz  )=>(T=0,T_K=0, Tnew=0, T_init=0, T_it_old=0, Tupdate=0, Tbuffer=0, Kc=1, Rho=1, Cp=1, Hr=0, Hl=0, ϕ=0, dϕdT=0,dϕdT_o=0, R=0, Z=0, P=0),
                                    (Num.Nx-1,Num.Nz  )=>(qx=0,Kx=0, Rc=0),
                                    (Num.Nx  ,Num.Nz-1)=>(qz=0,Kz=0 )
                                    ))
    else
        Arrays = CreateArrays3D(Dict( (Num.Nx,  Num.Ny  , Num.Nz  )=>(T=0,T_K=0, Tnew=0, T_init=0, T_it_old=0, Tupdate=0, Tbuffer=0, Kc=1, Rho=1, Cp=1, Hr=0, Hl=0, ϕ=0, dϕdT=0,dϕdT_o=0, R=0, X=0, Y=0, Z=0, P=0),
                                    (Num.Nx-1,Num.Ny  , Num.Nz  )=>(qx=0,Kx=0),
                                    (Num.Nx  ,Num.Ny-1, Num.Nz  )=>(qy=0,Ky=0),
                                    (Num.Nx  ,Num.Ny  , Num.Nz-1)=>(qz=0,Kz=0 )
                                    ))
    end

    return Arrays
end

"""
    MTK_initialize!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructArray, Dikes::SillParameters, CartData_input::CartData)

Initialize temperature and phases
"""
function MTK_initialize!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructArray, Dikes::SillParameters, CartData_input::Union{Nothing,CartData})
    # Initalize T from CartData set
    # NOTE: this almost certainly requires changes if we use GPUs

    if Num.USE_GPU
        if Num.dim==2
            Arrays.T_init       .= DataArray(CartData_input.fields.Temp[:,:,1])
            Arrays.Phases       .= DataArray(CartData_input.fields.Phases[:,:,1]);
            Arrays.Phases_init  .= DataArray(CartData_input.fields.Phases[:,:,1]);
        else
            Arrays.T_init       .= DataArray(CartData_input.fields.Temp)
            Arrays.Phases       .= DataArray(CartData_input.fields.Phases);
            Arrays.Phases_init  .= DataArray(CartData_input.fields.Phases);
        end
    else
        if Num.dim==2
            Arrays.T_init       .= CartData_input.fields.Temp[:,:,1];
            Arrays.Phases       .= CartData_input.fields.Phases[:,:,1];
            Arrays.Phases_init  .= CartData_input.fields.Phases[:,:,1];
        else
            Arrays.T_init       .= CartData_input.fields.Temp;
            Arrays.Phases       .= CartData_input.fields.Phases;
            Arrays.Phases_init  .= CartData_input.fields.Phases;
        end
    end

    # open pvd file if requested
    if Num.Output_VTK
        name =  joinpath(Num.SimName,Num.SimName*".pvd")
        Num.pvd = movie_paraview(name=name, Initialize=true);
    end

    return nothing
end


"""
    MTK_finalize!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructArray, Dikes::SillParameters, CartData_input::CartData)

Finalize model run
"""
function MTK_finalize!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructArray, Dikes::SillParameters, CartData_input::Union{Nothing,CartData})
    if Num.Output_VTK & !isnothing(Num.pvd)
        movie_paraview(pvd=Num.pvd, Finalize=true)
    end

    return nothing
end


"""
    MTK_update_Arrays!(Arrays::NamedTuple, Grid::GridData, Dikes::SillParameters, Num::NumericalParameters, Mat_tup::Tuple)

Update arrays and structs of the simulation (in case you want to change them during a simulation)
You can use this, for example, to change the size and location of an intruded dike
"""
function MTK_update_ArraysStructs!(Arrays::NamedTuple, Grid::GridData, Dikes::SillParameters, Num::NumericalParameters, Mat_tup::Tuple)

    if Num.AddRandomSills && mod(Num.it,Num.RandomSills_timestep)==0
        # This randomly changes the location and orientation of the sills
        if Num.dim==2
            Loc = [Dikes.W_ran; Dikes.H_ran]
        else
            Loc = [Dikes.W_ran; Dikes.L_ran; Dikes.H_ran]
        end

        # Randomly change location of center of dike/sill
        cen       = (Grid.max .+ Grid.min)./2 .+ rand(-0.5:1e-3:0.5, Num.dim).*Loc;

        Dip       = rand(-Dikes.Dip_ran/2.0    :   0.1:   Dikes.Dip_ran/2.0)
        Strike    = rand(-Dikes.Strike_ran/2.0 :   0.1:   Dikes.Strike_ran/2.0)

        if cen[end]<Dikes.SillsAbove;
            Dip = Dip   + 90.0                                          # Orientation: near-vertical @ depth
        end

        sill = _active_sill(Dikes)
        if Num.dim == 2
            Dikes.sill = InjectSills.update_abstractsill(sill;
                                                         Center=InjectSills.Point2(cen[1], cen[2]) * m,
                                                         Angle=InjectSills.Vec1(Dip) * NoUnits)
        else
            Dikes.sill = InjectSills.update_abstractsill(sill;
                                                         Center=InjectSills.Point3(cen[1], cen[2], cen[3]) * m,
                                                         Angle=InjectSills.Vec2(Dip, Strike) * NoUnits)
        end
    end
    return nothing
end


"""
    MTK_save_output(Grid::GridData, Arrays::NamedTuple, Tracers::StructArray, Dikes::SillParameters, time_props::TimeDependentProperties, Num::NumericalParameters, CartData_input::Union{CartData, Nothing})

Save the output to disk
"""
function MTK_save_output(Grid::GridData, Arrays::NamedTuple, Tracers::StructArray, Dikes::SillParameters, time_props::TimeDependentProperties, Num::NumericalParameters, CartData_input::Union{CartData, Nothing})

    if mod(Num.it,Num.SaveOutput_steps)==0
        # Save output
        if Num.Output_VTK
            name = joinpath(Num.SimName,Num.SimName*"_$(Num.it)")
            if !isnothing(CartData_input)
                Data_set3D  = CartData_input
            else
                if length(Grid.coord1D)==3
                    X,Y,Z   =   xyz_grid(Grid.coord1D...)
                elseif length(Grid.coord1D)==2
                    X,Y,Z   =   xyz_grid(Grid.coord1D[1], 0, Grid.coord1D[2])
                end
                Data_set3D  =   CartData(X/1e3,Y/1e3,Z/1e3, (Z=Z,))
            end
            # add datasets
            Data_set3D = add_data_CartData(Data_set3D, "Temp",         Float32.(Array(Arrays.Tnew)));
            Data_set3D = add_data_CartData(Data_set3D, "Phases",       Int32.(Array(Arrays.Phases)));
            Data_set3D = add_data_CartData(Data_set3D, "MeltFraction", Float64.(Array(Arrays.ϕ)));

            # Save output to CartData
            Num.pvd  = write_paraview(Data_set3D, name, pvd=Num.pvd,time=Num.time/SecYear/1e3);
        end
    end
    return nothing
end


"""
    d = add_data_CartData(d::CartData, name::String, data::Array)
Adds data from MTK to a CartData structure, both in 2D & 3D
"""
function add_data_CartData(d::CartData, name::String, data::Array)
    if length(size(data)) == 2
        a = zero(d.x.val)
        if size(a)[3]==1
            a[:,:,1] .= data;
        elseif size(a)[2]==1
            a[:,1,:] .= data;
        end
    else
        a = data
    end
    d = addfield(d, name, a)
    return d
end


"""
    Tracers = MTK_updateTracers(Grid::GridData, Arrays::NamedTuple, Tracers::StructArray, Dikes::SillParameters, time_props::TimeDependentProperties, Num::NumericalParameters)

Updates info on tracers
"""
function MTK_updateTracers(Grid::GridData, Arrays::NamedTuple, Tracers::StructArray, Dikes::SillParameters, time_props::TimeDependentProperties, Num::NumericalParameters)

    if mod(Num.it,10)==0
        update_Tvec!(Tracers, Num.time/SecYear*1e-6)  # update T & time vectors on tracers
    end

    return Tracers
end

"""
    Num = Setup_Model_CartData(d::CartData, Num::NumericalParameters, Mat_tup::Tuple)

Create a MTK model setup from a CartData structure generated with GeophysicalModelGenerator

"""
function Setup_Model_CartData(d::CartData, Num::NumericalParameters, Mat_tup::Tuple)
    if size(d.x)[3] == 1
        Num = Setup_Model_CartData_2D(d, Num, Mat_tup)
    else
        Num = Setup_Model_CartData_3D(d, Num, Mat_tup)
    end
    return Num
end


function Setup_Model_CartData_2D(d::CartData, Num::NumericalParameters, Mat_tup::Tuple)
    @assert size(d.x)[3] == 1
    x = extrema(d.fields.FlatCrossSection.*1e3)
    z = extrema(d.z.val.*1e3)

    Num.W = (x[2]-x[1])
    Num.H = (z[2]-z[1])
    Num.Nx = size(d.x)[1]
    Num.Nz = size(d.x)[2]

    dx = (x[2]-x[1])/(Num.Nx-1)
    dz = (z[2]-z[1])/(Num.Nz-1)

    # estimate maximum thermal diffusivity from Mat_tup
    κ_max = Num.κ_time
    for mm in Mat_tup
        if hasfield(typeof(mm.Conductivity[1]),:k)
            k = NumValue(mm.Conductivity[1].k)
        else
            k = 3;
        end
        if hasfield(typeof(mm.HeatCapacity[1]),:cp)
            cp = NumValue(mm.HeatCapacity[1].cp)
        else
            cp = 1050;
        end
        if hasfield(typeof(mm.Density[1]),:ρ)
            ρ = NumValue(mm.Density[1].ρ)
        else
            ρ = 2700;
        end
        κ  = k/(cp*ρ)
        if κ>κ_max
            κ_max = κ
        end
    end
    Num.κ_time = κ_max;
    Num.Δ = [dx, dz]
    Num.Δmin  =   minimum(Num.Δ[Num.Δ.>0]);               # minimum grid spacing

    Num.dt = Num.fac_dt*(Num.Δmin^2)./Num.κ_time/4;   # timestep

    Num.dx = dx;
    Num.dz = dz;

    Num.nt = floor(Num.maxTime/Num.dt)

    return Num
end

function Setup_Model_CartData_3D(d::CartData, Num::NumericalParameters, Mat_tup::Tuple)
    x = extrema(d.x.val.*1e3)
    y = extrema(d.y.val.*1e3)
    z = extrema(d.z.val.*1e3)

    Num.W = (x[2]-x[1])
    Num.L = (y[2]-y[1])
    Num.H = (z[2]-z[1])
    Num.Nx = size(d.x)[1]
    Num.Ny = size(d.x)[2]
    Num.Nz = size(d.x)[3]

    dx = (x[2]-x[1])/(Num.Nx-1)
    dy = (y[2]-y[1])/(Num.Ny-1)
    dz = (z[2]-z[1])/(Num.Nz-1)

    # estimate maximum thermal diffusivity from Mat_tup
    κ_max = Num.κ_time
    for mm in Mat_tup
        if hasfield(typeof(mm.Conductivity[1]),:k)
            k = NumValue(mm.Conductivity[1].k)
        else
            k = 3;
        end
        if hasfield(typeof(mm.HeatCapacity[1]),:cp)
            cp = NumValue(mm.HeatCapacity[1].cp)
        else
            cp = 1050;
        end
        if hasfield(typeof(mm.Density[1]),:ρ)
            ρ = NumValue(mm.Density[1].ρ)
        else
            ρ = 2700;
        end
        κ  = k/(cp*ρ)
        if κ>κ_max
            κ_max = κ
        end
    end
    Num.κ_time = κ_max;
    Num.Δ = [dx, dy, dz]
    Num.Δmin  =   minimum(Num.Δ[Num.Δ.>0]);               # minimum grid spacing

    Num.dt = Num.fac_dt*(Num.Δmin^2)./Num.κ_time/4;   # timestep
    Num.dx = dx;
    Num.dy = dy;
    Num.dz = dz;

    Num.nt = floor(Num.maxTime/Num.dt)

    return Num
end



end
