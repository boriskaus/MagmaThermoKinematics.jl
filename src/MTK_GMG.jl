# various routines that are shared between the 2D and 3D MTK_GMG routines
"""
    MTK_GMG
This contains various user callback routines that are shared between the 2D and 3D MTK_GMG routines.
You can overwrite this in your own code to customize the simulation.

"""
module MTK_GMG

using Parameters
using GeoParams
using GeophysicalModelGenerator
using StructArrays
using MagmaThermoKinematics.Grid
using MagmaThermoKinematics.Data
import MagmaThermoKinematics.Fields2D: CreateArrays as CreateArrays2D
import MagmaThermoKinematics.Fields3D: CreateArrays as CreateArrays3D
import MagmaThermoKinematics: NumericalParameters, DikeParameters, TimeDependentProperties
import MagmaThermoKinematics: update_Tvec!, Dike, InjectDike, km³, kyr, Myr, CreateDikePolygon
import MagmaThermoKinematics: PhasesFromTracers!
SecYear = 3600*24*365.25;

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
function MTK_inject_dikes(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::DikeParameters, Tracers::StructVector, Tnew_cpu)

    if floor(Num.time/Dikes.InjectionInterval)> Dikes.dike_inj
        Dikes.dike_inj      =   floor(Num.time/Dikes.InjectionInterval)                 # Keeps track on what was injected already
        if Num.dim==2
            T_bottom  =   Tnew_cpu[:,1]
        else
            T_bottom  =   Tnew_cpu[:,:,1]
        end
        dike      =   Dike(W=Dikes.W_in, H=Dikes.H_in, Type=Dikes.Type, T=Dikes.T_in_Celsius, Center=Dikes.Center[:],  Angle=Dikes.Angle, Phase=Dikes.DikePhase);               # "Reference" dike with given thickness,radius and T
        Tnew_cpu .=   Array(Arrays.T)

        Tracers, Tnew_cpu,Vol,Dikes.dike_poly, VEL  =   InjectDike(Tracers, Tnew_cpu, Grid.coord1D, dike, Dikes.nTr_dike, dike_poly=Dikes.dike_poly);     # Add dike, move hostrocks

        if Num.flux_bottom_BC==false
            # Keep bottom T constant (advection modifies this)
            if Num.dim==2
                Tnew_cpu[:,1]     .=  T_bottom
            else
                Tnew_cpu[:,:,1]   .=  T_bottom
            end
        end

        Arrays.T           .=   Data.Array(Tnew_cpu)
        Dikes.InjectVol    +=   Vol                                                     # Keep track of injected volume
        Qrate               =   Dikes.InjectVol/Num.time
        Dikes.Qrate_km3_yr  =   Qrate*SecYear/km³
        Qrate_km3_yr_km2    =   Dikes.Qrate_km3_yr/(pi*(Dikes.W_in/2/1e3)^2)
        println("  Added new dike; time=$(Num.time/kyr) kyrs, total injected magma volume = $(Dikes.InjectVol/km³) km³; rate Q= $(Dikes.Qrate_km3_yr) km³yr⁻¹")

        if Num.advect_polygon==true && isempty(Dikes.dike_poly)
            Dikes.dike_poly   =   CreateDikePolygon(dike);            # create dike for the 1th time
        end

        if length(Mat_tup)>1
           PhasesFromTracers!(Array(Arrays.Phases), Grid, Tracers, BackgroundPhase=Dikes.BackgroundPhase, InterpolationMethod="Constant");    # update phases from grid

           # Ensure that we keep the initial phase of the area (host rocks are not deformable)
           if Num.keep_init_RockPhases==true
                Phases      = Array(Arrays.Phases)          # move to CPU
                Phases_init = Array(Arrays.Phases_init)
                for i in eachindex(Phases)
                    if Phases[i] != Dikes.DikePhase
                        Phases[i] = Phases_init[i]
                    end
                end
                Arrays.Phases .= Data.Array(Phases)          # move back to GPU
           end
        end

    end

    return Tracers
end

"""
    MTK_display_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::DikeParameters)

Function that creates plots
"""
function MTK_visualize_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::DikeParameters)

    return nothing
end

"""
    MTK_print_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::DikeParameters)

Function that prints output to the REPL
"""
function MTK_print_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::DikeParameters)

    return nothing
end

"""
    MTK_update_TimeDepProps!(time_props::TimeDependentProperties, Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::DikeParameters)

Update time-dependent properties during a simulation
"""
function MTK_update_TimeDepProps!(time_props::TimeDependentProperties, Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::DikeParameters)
    push!(time_props.Time_vec,      Num.time);   # time
    push!(time_props.MeltFraction,  sum( Arrays.ϕ)/(Num.Nx*Num.Nz));    # melt fraction

    ind = findall(Arrays.T.>700);
    if ~isempty(ind)
        Tav_magma_Time = sum(Arrays.T[ind])/length(ind)     # average T of part with magma
    else
        Tav_magma_Time = NaN;
    end
    push!(time_props.Tav_magma, Tav_magma_Time);       # average magma T
    push!(time_props.Tmax,      maximum(Arrays.T));   # maximum magma T

    return nothing
end

"""
    MTK_initialize!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructArray, Dikes::DikeParameters)

Initialize temperature and phases
"""
function MTK_initialize!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructArray, Dikes::DikeParameters)
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
        Arrays = CreateArrays2D(Dict( (Num.Nx,  Num.Nz  )=>(T=0,T_K=0, Tnew=0, T_init=0, T_it_old=0, Kc=1, Rho=1, Cp=1, Hr=0, Hl=0, ϕ=0, dϕdT=0,dϕdT_o=0, R=0, Z=0, P=0),
                                    (Num.Nx-1,Num.Nz  )=>(qx=0,Kx=0, Rc=0),
                                    (Num.Nx  ,Num.Nz-1)=>(qz=0,Kz=0 )
                                    ))
    else
        Arrays = CreateArrays3D(Dict( (Num.Nx,  Num.Ny  , Num.Nz  )=>(T=0,T_K=0, Tnew=0, T_init=0, T_it_old=0, Kc=1, Rho=1, Cp=1, Hr=0, Hl=0, ϕ=0, dϕdT=0,dϕdT_o=0, R=0, X=0, Y=0, Z=0, P=0),
                                    (Num.Nx-1,Num.Ny  , Num.Nz  )=>(qx=0,Kx=0),
                                    (Num.Nx  ,Num.Ny-1, Num.Nz  )=>(qy=0,Ky=0),
                                    (Num.Nx  ,Num.Ny  , Num.Nz-1)=>(qz=0,Kz=0 )
                                    ))
    end

    return Arrays
end

"""
    MTK_initialize!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructArray, Dikes::DikeParameters, CartData_input::CartData)

Initialize temperature and phases
"""
function MTK_initialize!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructArray, Dikes::DikeParameters, CartData_input::Union{Nothing,CartData})
    # Initalize T from CartData set
    # NOTE: this almost certainly requires changes if we use GPUs

    if Num.USE_GPU
        if Num.dim==2
            Arrays.T_init       .= Data.Array(CartData_input.fields.Temp[:,:,1])
            Arrays.Phases       .= Data.Array(CartData_input.fields.Phases[:,:,1]);
            Arrays.Phases_init  .= Data.Array(CartData_input.fields.Phases[:,:,1]);
        else
            Arrays.T_init       .= Data.Array(CartData_input.fields.Temp)
            Arrays.Phases       .= Data.Array(CartData_input.fields.Phases);
            Arrays.Phases_init  .= Data.Array(CartData_input.fields.Phases);
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
    MTK_finalize!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructArray, Dikes::DikeParameters, CartData_input::CartData)

Finalize model run
"""
function MTK_finalize!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructArray, Dikes::DikeParameters, CartData_input::Union{Nothing,CartData})
    if Num.Output_VTK & !isnothing(Num.pvd)
        movie_paraview(pvd=Num.pvd, Finalize=true)
    end

    return nothing
end


"""
    MTK_update_Arrays!(Arrays::NamedTuple, Grid::GridData, Dikes::DikeParameters, Num::NumericalParameters, Mat_tup::Tuple)

Update arrays and structs of the simulation (in case you want to change them during a simulation)
You can use this, for example, to change the size and location of an intruded dike
"""
function MTK_update_ArraysStructs!(Arrays::NamedTuple, Grid::GridData, Dikes::DikeParameters, Num::NumericalParameters, Mat_tup::Tuple)

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

        Dikes.Center = cen;
        Dikes.Angle  = [Dip, Strike];
    end
    return nothing
end


"""
    MTK_save_output(Grid::GridData, Arrays::NamedTuple, Tracers::StructArray, Dikes::DikeParameters, time_props::TimeDependentProperties, Num::NumericalParameters, CartData_input::Union{CartData, Nothing})

Save the output to disk
"""
function MTK_save_output(Grid::GridData, Arrays::NamedTuple, Tracers::StructArray, Dikes::DikeParameters, time_props::TimeDependentProperties, Num::NumericalParameters, CartData_input::Union{CartData, Nothing})

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
            Data_set3D = add_data_CartData(Data_set3D, "Temp",         Array(Arrays.Tnew));
            Data_set3D = add_data_CartData(Data_set3D, "Phases",       Array(Arrays.Phases));
            Data_set3D = add_data_CartData(Data_set3D, "MeltFraction", Array(Arrays.ϕ));

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
    Tracers = MTK_updateTracers(Grid::GridData, Arrays::NamedTuple, Tracers::StructArray, Dikes::DikeParameters, time_props::TimeDependentProperties, Num::NumericalParameters)

Updates info on tracers
"""
function MTK_updateTracers(Grid::GridData, Arrays::NamedTuple, Tracers::StructArray, Dikes::DikeParameters, time_props::TimeDependentProperties, Num::NumericalParameters)

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
