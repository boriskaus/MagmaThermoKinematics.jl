# This contains the 2D routine to create MTK simulations (using GeoParams)

#
module MTK_GMG_2D
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using Parameters
using StructArrays
using GeophysicalModelGenerator

__init__() = @init_parallel_stencil(Threads, Float64, 2)

using MagmaThermoKinematics.Diffusion2D
using MagmaThermoKinematics.MTK_GMG
using MagmaThermoKinematics

const SecYear = 3600*24*365.25;

export MTK_GeoParams_2D

#-----------------------------------------------------------------------------------------
"""
    Grid, Arrays, Tracers, Dikes, time_props = MTK_GeoParams_2D(Mat_tup::Tuple, Num::NumericalParameters, Dikes::DikeParameters; CartData_input=nothing, time_props::TimeDependentProperties = TimeDepProps());

Main routine that performs a 2D or 2D axisymmetric thermal diffusion simulation with injection of dikes.

Parameters
====
- `Mat_tup::Tuple`: Tuple of material properties.
- `Num::NumericalParameters`: Numerical parameters.
- `Dikes::DikeParameters`: Dike parameters.
- `CartData_input::CartData`: Optional input of a CartData structure generated with GeophysicalModelGenerator.
- `time_props::TimeDependentProperties`: Optional input of a `TimeDependentProperties` structure.

Customizable functions
====
There are a few functions that you can overwrite in your user code to customize the simulation:

- `MTK_visualize_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::DikeParameters)`
- `MTK_update_TimeDepProps!(time_props::TimeDependentProperties, Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::DikeParameters)`
- `MTK_update_ArraysStructs!(Arrays::NamedTuple, Grid::GridData, Dikes::DikeParameters, Num::NumericalParameters)`
- `MTK_initialize!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructArray, Dikes::DikeParameters, CartData_input)`
- `MTK_updateTracers(Grid::GridData, Arrays::NamedTuple, Tracers::StructArray, Dikes::DikeParameters, time_props::TimeDependentProperties, Num::NumericalParameters)`
- `MTK_save_output(Grid::GridData, Arrays::NamedTuple, Tracers::StructArray, Dikes::DikeParameters, time_props::TimeDependentProperties, Num::NumericalParameters, CartData_input::CartData)`
- `MTK_inject_dikes(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::DikeParameters, Tracers::StructVector, Tnew_cpu)`
- `MTK_initialize!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructArray, Dikes::DikeParameters)`
- `MTK_finalize!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructArray, Dikes::DikeParameters, CartData_input::CartData)`

"""
@views function MTK_GeoParams_2D(Mat_tup::Tuple, Num::NumericalParameters, Dikes::DikeParameters; CartData_input::Union{Nothing,CartData}=nothing, time_props::TimeDependentProperties = TimeDepProps());

    # Change parameters based on CartData input
    Num.dim = 2;
    if !isnothing(CartData_input)

        if !hasfield(typeof(CartData_input.fields),:FlatCrossSection)
           error("You should add a Field :FlatCrossSection to your data structure with Data_Cross = addfield(Data_Cross,\"FlatCrossSection\", flatten_cross_section(Data_Cross))")
        end

        Num = MTK_GMG.Setup_Model_CartData(CartData_input, Num, Mat_tup)
    end

    # Array & grid initializations ---------------
    Arrays = MTK_GMG.MTK_initialize_arrays(Num);

    # Set up model geometry & initial T structure
    if isnothing(CartData_input)
        Grid    = CreateGrid(size=(Num.Nx,Num.Nz), extent=(Num.W, Num.H))
    else
        Grid    = CreateGrid(CartData_input)
    end
    GridArray!(Arrays.R, Arrays.Z, Grid)
    Arrays.Rc              .=   (Arrays.R[2:end,:] + Arrays.R[1:end-1,:])/2     # center points in x
    # --------------------------------------------

    Tracers                 =   StructArray{Tracer}(undef, 1)                       # Initialize tracers

    # Update buffer & phases arrays --------------
    if Num.USE_GPU
        # CPU buffers for advection
        Tnew_cpu        =   Matrix{Float64}(undef, Num.Nx, Num.Nz)
        Phi_melt_cpu    =   similar(Tnew_cpu)
        Phases          =   CUDA.ones(Int64,Num.Nx,Num.Nz)
        Phases_init     =   CUDA.ones(Int64,Num.Nx,Num.Nz)
    else
        Tnew_cpu        =   similar(Arrays.T)
        Phi_melt_cpu    =   similar(Arrays.ϕ)
        Phases          =   ones(Int64,Num.Nx,Num.Nz)
        Phases_init     =   ones(Int64,Num.Nx,Num.Nz)
    end
    Arrays = (Arrays..., Phases=Phases, Phases_init=Phases_init);

    # Initialize Geotherm and Phases -------------
    if isnothing(CartData_input)
        MTK_GMG.MTK_initialize!(Arrays, Grid, Num, Tracers, Dikes);
    else
        MTK_GMG.MTK_initialize!(Arrays, Grid, Num, Tracers, Dikes, CartData_input);
    end
    # --------------------------------------------

    # check errors
    unique_Phases = unique(Array(Arrays.Phases));
    phase_specified = []
    for mm in Mat_tup
        push!(phase_specified, mm.Phase)
    end
    for u in unique_Phases
        if !(u in phase_specified)
            error("Properties for Phase $u are not specified in Mat_tup. Please add that")
        end
    end

    if any(isnan.(Arrays.T))
        error("NaNs in T; something is wrong")
    end

    # Optionally set initial sill in models ------
    if Dikes.Type  == "CylindricalDike_TopAccretion"
        ind = findall( (Arrays.R.<=Dikes.W_in/2) .& (abs.(Arrays.Z.-Dikes.Center[2]) .< Dikes.H_in/2) );
        Arrays.T_init[ind] .= Dikes.T_in_Celsius;
        if Num.advect_polygon==true
            dike              =   Dike(W=Dikes.W_in,H=Dikes.H_in,Type=Dikes.Type,T=Dikes.T_in_Celsius, Center=Dikes.Center[:],  Angle=Dikes.Angle, Phase=Dikes.DikePhase);               # "Reference" dike with given thickness,radius and T
            Dikes.dike_poly   =   CreateDikePolygon(dike);
        end
    end
    # --------------------------------------------

    # Initialise arrays --------------------------
    @parallel assign!(Arrays.Tnew, Arrays.T_init)
    @parallel assign!(Arrays.T, Arrays.T_init)

    if isdir(Num.SimName)==false
        mkdir(Num.SimName)          # create simulation directory if needed
    end;
    # --------------------------------------------

    for Num.it = 1:Num.nt   # Time loop
        Num.time  += Num.dt;                                     # Keep track of evolved time

        # Add new dike every X years -----------------
        Tracers = MTK_GMG.MTK_inject_dikes(Grid, Num, Arrays, Mat_tup, Dikes, Tracers, Tnew_cpu)
        # --------------------------------------------

        # Do a diffusion step, while taking T-dependencies into account
        Nonlinear_Diffusion_step_2D!(Arrays, Mat_tup, Phases, Grid, Num.dt, Num)
        # --------------------------------------------

        # Update variables ---------------------------
        # copy to cpu
        Tnew_cpu      .= Data.Array(Arrays.Tnew)
        Phi_melt_cpu  .= Data.Array(Arrays.ϕ)

        UpdateTracers_T_ϕ!(Tracers, Grid.coord1D, Tnew_cpu, Phi_melt_cpu);     # Update info on tracers

        # copy back to gpu
        Arrays.Tnew   .= Data.Array(Tnew_cpu)
        Arrays.ϕ      .= Data.Array(Phi_melt_cpu)

        @parallel assign!(Arrays.T, Arrays.Tnew)
        @parallel assign!(Arrays.Tnew, Arrays.T)
        # --------------------------------------------

        # Update info on tracers ---------------------
        Tracers = MTK_GMG.MTK_updateTracers(Grid, Arrays, Tracers, Dikes, time_props, Num);
        # --------------------------------------------

        # Update time-dependent properties -----------
        MTK_GMG.MTK_update_TimeDepProps!(time_props, Grid, Num, Arrays, Mat_tup, Dikes)
        # --------------------------------------------

        # Visualize results --------------------------
        MTK_GMG.MTK_visualize_output(Grid, Num, Arrays, Mat_tup, Dikes)
        # --------------------------------------------

        # Save output to disk once in a while --------
        MTK_GMG.MTK_save_output(Grid, Arrays, Tracers, Dikes, time_props, Num, CartData_input);
        # --------------------------------------------

        # Optionally update arrays and structs (such as T or Dike) -------
        MTK_GMG.MTK_update_ArraysStructs!(Arrays, Grid, Dikes, Num, Mat_tup)
        # --------------------------------------------

        # Display output -----------------------------
        MTK_GMG.MTK_print_output(Grid, Num, Arrays, Mat_tup, Dikes)
        # --------------------------------------------

    end

    # Finalize simulation ------------------------
    MTK_GMG.MTK_finalize!(Arrays, Grid, Num, Tracers, Dikes, CartData_input);
    # --------------------------------------------

    return Grid, Arrays, Tracers, Dikes, time_props
end # end of main function

end
