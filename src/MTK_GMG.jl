"""
    MTK_GMG
This contains the user callback routines that [`MTK_GeoParams`](@ref) calls in 2D and 3D.
You can overwrite them in your own code to customize the simulation.

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
import MagmaThermoKinematics: PhasesFromTracers!, CreateArrays, copy_to_device!
SecYear = 3600*24*365.25;

@inline _active_sill(Dikes) = isnothing(Dikes.sill) ? error("SillParameters requires a valid `sill` object") : Dikes.sill

"""
    Analytical geotherm used for the UCLA setups, which includes radioactive heating
"""
function AnalyticalGeotherm!(T, Z, Tsurf, qm, qs, k, hr)
    FT = eltype(T)
    Tsurf, qm, qs, k, hr = FT(Tsurf), FT(qm), FT(qs), FT(k), FT(hr)
    T      .=  @. Tsurf - (qm/k)*Z + (qs-qm)*hr/k*( one(FT) - exp(Z/hr))
    return nothing
end

"""
    Tracers = MTK_inject_dikes(Grid, Num, Arrays, Mat_tup, Dikes, Tracers)

Function that injects dikes once in a while
"""
function MTK_inject_dikes(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters, Tracers::StructVector)

    if floor(Num.time/Dikes.InjectionInterval)> Dikes.sill_inj
        Dikes.sill_inj = floor(Num.time/Dikes.InjectionInterval)                 # Keeps track on what was injected already
        T_bottom  =   copy(selectdim(Arrays.T, Num.dim, 1))
        sill = _active_sill(Dikes)

        Tracers, _, Vol, poly_out, _ = inject_sills(Tracers, Arrays.T, Grid.coord1D, sill, Float64(Dikes.T_in_Celsius), Dikes.SillPhase, Dikes.nTr_dike, dike_poly=Dikes.sill_poly);     # Add dike, move hostrocks
        Dikes.sill_poly = poly_out

        if Num.flux_bottom_BC==false
            # Keep bottom T constant (advection modifies this)
            selectdim(Arrays.T, Num.dim, 1) .= T_bottom
        end

        Dikes.InjectVol    +=   Vol                                                     # Keep track of injected volume
        Qrate               =   Dikes.InjectVol/Num.time
        Dikes.Qrate_km3_yr  =   Qrate*SecYear/km³
        println("  Added new dike; time=$(Num.time/kyr) kyrs, total injected magma volume = $(Dikes.InjectVol/km³) km³; rate Q= $(Dikes.Qrate_km3_yr) km³yr⁻¹")

        if Num.advect_polygon==true && isempty(Dikes.sill_poly)
            Dikes.sill_poly = InjectSills.dike_polygon(sill)            # create sill polygon for the first time
        end

        if length(Mat_tup)>1
            Phases = Array(Arrays.Phases)
            PhasesFromTracers!(Phases, Grid, Tracers, BackgroundPhase=Dikes.BackgroundPhase, InterpolationMethod="Constant");    # update phases from tracers

            # Ensure that we keep the initial phase of the area (host rocks are not deformable)
            if Num.keep_init_RockPhases==true
                Phases_init = Array(Arrays.Phases_init)
                for i in eachindex(Phases, Phases_init)
                    if Phases[i] != Dikes.SillPhase
                        Phases[i] = Phases_init[i]
                    end
                end
            end
            copyto!(Arrays.Phases, Phases)
        end

    end

    return Tracers
end

"""
    MTK_visualize_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters)

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
    push!(time_props.MeltFraction,  sum(Arrays.ϕ)/length(Arrays.ϕ));    # mean melt fraction

    n_hot = count(>(700), Arrays.T)
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
    FT = eltype(Arrays.T_init)
    Tsurf, Geotherm = FT(Num.Tsurface_Celcius), FT(Num.Geotherm)
    Arrays.T_init      .=   @. Tsurf - Arrays.Z*Geotherm;                # Initial (linear) temperature profile

    # Open pvd file if requested
    if Num.Output_VTK
        name =  joinpath(Num.SimName,Num.SimName*".pvd")
        Num.pvd = movie_paraview(name=name, Initialize=true);
    end

    return nothing
end

"""
    Arrays = MTK_initialize_arrays(Num::NumericalParameters)

Initialize arrays used in the computations
"""
function MTK_initialize_arrays(Num::NumericalParameters)

    kw = (; backend=Num.backend, FloatType=Num.FloatType)
    if Num.dim==2
        Arrays = CreateArrays(Dict( (Num.Nx,  Num.Nz  )=>(T=0,T_K=0, Tnew=0, T_init=0, T_it_old=0, Tupdate=0, Kc=1, Rho=1, Cp=1, Hr=0, Hl=0, ϕ=0, dϕdT=0, R=0, Z=0, P=0)); kw...)
    else
        Arrays = CreateArrays(Dict( (Num.Nx,  Num.Ny  , Num.Nz  )=>(T=0,T_K=0, Tnew=0, T_init=0, T_it_old=0, Tupdate=0, Kc=1, Rho=1, Cp=1, Hr=0, Hl=0, ϕ=0, dϕdT=0, X=0, Y=0, Z=0, P=0)); kw...)
    end

    return Arrays
end

"""
    MTK_initialize!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructArray, Dikes::SillParameters, CartData_input::CartData)

Initialize temperature and phases
"""
function MTK_initialize!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructArray, Dikes::SillParameters, CartData_input::Union{Nothing,CartData})
    # Initalize T and phases from the CartData set
    if Num.dim==2
        Temp, Phases = CartData_input.fields.Temp[:,:,1], CartData_input.fields.Phases[:,:,1]
    else
        Temp, Phases = CartData_input.fields.Temp, CartData_input.fields.Phases
    end
    copy_to_device!(Arrays.T_init, Temp)
    copy_to_device!(Arrays.Phases, Phases)
    copy_to_device!(Arrays.Phases_init, Phases)

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
