# This contains the 2D routine to create MTK simulations (using GeoParams)

#
module MTK_GMG_2D
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using Parameters
using StructArrays
using GeophysicalModelGenerator

@init_parallel_stencil(Threads, Float64, 2)

import ..Diffusion2D: GridArray!, Nonlinear_Diffusion_step_2D!, assign!
using ..MTK_GMG
import ..NumericalParameters, ..SillParameters, ..TimeDependentProperties, ..TimeDepProps
import ..EruptionParameters, ..EruptionParams
import ..FreeSurfaceParameters, ..FreeSurfaceParams
import ..CreateGrid, ..Tracer, ..UpdateTracers_T_ϕ!, ..InjectSills, ..m, ..NoUnits
import ..seed_host_tracers

const SecYear = 3600*24*365.25;

export MTK_GeoParams_2D

#-----------------------------------------------------------------------------------------
"""
    Grid, Arrays, Tracers, Dikes, time_props = MTK_GeoParams_2D(Mat_tup::Tuple, Num::NumericalParameters, Dikes::SillParameters; CartData_input=nothing, time_props::TimeDependentProperties = TimeDepProps());

Main routine that performs a 2D or 2D axisymmetric thermal diffusion simulation with injection of dikes.

Parameters
====
- `Mat_tup::Tuple`: Tuple of material properties.
- `Num::NumericalParameters`: Numerical parameters.
- `Dikes::SillParameters`: Intrusion parameters.
- `CartData_input::CartData`: Optional input of a CartData structure generated with GeophysicalModelGenerator.
- `time_props::TimeDependentProperties`: Optional input of a `TimeDependentProperties` structure.

Customizable functions
====
There are a few functions that you can overwrite in your user code to customize the simulation:

- `MTK_visualize_output(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters)`
- `MTK_update_TimeDepProps!(time_props::TimeDependentProperties, Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters)`
- `MTK_update_ArraysStructs!(Arrays::NamedTuple, Grid::GridData, Dikes::SillParameters, Num::NumericalParameters)`
- `MTK_initialize!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructArray, Dikes::SillParameters, CartData_input)`
- `MTK_updateTracers(Grid::GridData, Arrays::NamedTuple, Tracers::StructArray, Dikes::SillParameters, time_props::TimeDependentProperties, Num::NumericalParameters)`
- `MTK_save_output(Grid::GridData, Arrays::NamedTuple, Tracers::StructArray, Dikes::SillParameters, time_props::TimeDependentProperties, Num::NumericalParameters, CartData_input::CartData)`
- `MTK_inject_dikes(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters, Tracers::StructVector, Tnew_cpu)`
- `MTK_erupt!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructVector, Erupt::EruptionParameters, FS::FreeSurfaceParameters, Mat_tup::Tuple, Dikes::SillParameters)`
- `MTK_free_surface!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Dikes::SillParameters, FS::FreeSurfaceParameters)`
- `MTK_initialize!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructArray, Dikes::SillParameters)`
- `MTK_finalize!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructArray, Dikes::SillParameters, CartData_input::CartData)`

"""
@views function MTK_GeoParams_2D(Mat_tup::Tuple, Num::NumericalParameters, Dikes::SillParameters; CartData_input::Union{Nothing,CartData}=nothing, time_props::TimeDependentProperties = TimeDepProps(), Erupt::EruptionParameters = EruptionParams(), FS::FreeSurfaceParameters = FreeSurfaceParams());

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

    Tracers                 =   StructArray{Tracer{Num.TracerFloatType}}(undef, 1)   # Initialize tracers

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
    if hasproperty(Dikes, :sill) && !isnothing(Dikes.sill) && Dikes.sill isa InjectSills.CylindricalDikeTopAccretion
        c = [Dikes.sill.Center[i].val for i in 1:2]
        ind = findall((Arrays.R .<= Dikes.sill.W.val) .& (abs.(Arrays.Z .- c[2]) .< Dikes.sill.H.val/2))
        Arrays.T_init[ind] .= Dikes.T_in_Celsius
        if Num.advect_polygon==true
            if hasproperty(Dikes, :sill_poly)
                Dikes.sill_poly = InjectSills.dike_polygon(Dikes.sill)
            else
                Dikes.dike_poly = InjectSills.dike_polygon(Dikes.sill)
            end
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

    # Initialise the free surface (if active) ----
    if FS.free_surface && isnothing(FS.z_surf)
        FS.z_surf = MTK_GMG.init_free_surface(Grid; z0=FS.z0, topography=FS.topography)
    end
    # A moving free surface needs a deformable phase field so the surface,
    # host rock and sills move together (injection inflation + eruption deflation).
    Num.deform_hostrock = Num.deform_hostrock || FS.free_surface
    # --------------------------------------------

    # Optionally seed passive host-rock tracers ---
    # They carry the initial layering and accumulate T-t paths; the existing
    # injection/deflation advection moves them and eruptions freeze them. The
    # phase field is still handled by advect_phases!.
    if Num.SeedHostTracers
        z_surf = FS.free_surface ? FS.z_surf : nothing
        Tracers = seed_host_tracers(Grid, Array(Arrays.Phases), Array(Arrays.T), Array(Arrays.ϕ);
                                    NumTracersDir=Num.HostTracersDir, air_phase=FS.air_phase, z_surf=z_surf)
    end
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
        # Copy fields only when tracers are active.
        if isassigned(Tracers,1)
            copyto!(Tnew_cpu, Arrays.Tnew)
            copyto!(Phi_melt_cpu, Arrays.ϕ)

            UpdateTracers_T_ϕ!(Tracers, Grid.coord1D, Tnew_cpu, Phi_melt_cpu);     # Update info on tracers
        end

        @parallel assign!(Arrays.T, Arrays.Tnew)
        # --------------------------------------------

        # Erupt magma when the eruptible volume reaches V_crit ----
        MTK_GMG.MTK_erupt!(Arrays, Grid, Num, Tracers, Erupt, FS, Mat_tup, Dikes)
        # --------------------------------------------

        # Update the moving free surface (inflation + air stamping) ----
        MTK_GMG.MTK_free_surface!(Arrays, Grid, Num, Dikes, FS)
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
