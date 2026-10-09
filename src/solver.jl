"""
    Grid, Arrays, Tracers, Dikes, time_props = MTK_GeoParams(Mat_tup::Tuple, Num::NumericalParameters, Dikes::SillParameters; CartData_input=nothing, time_props::TimeDependentProperties = TimeDepProps());

Main routine that performs a 2D, 2D axisymmetric or 3D thermal diffusion simulation with injection of dikes.
The model is 3D if `Num.Ny > 0` (or if `CartData_input` is 3D), and 2D otherwise.
The model arrays live on the KernelAbstractions backend `Num.backend`, with element type `Num.FloatType`.

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
- `MTK_update_ArraysStructs!(Arrays::NamedTuple, Grid::GridData, Dikes::SillParameters, Num::NumericalParameters, Mat_tup::Tuple)`
- `MTK_initialize!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructArray, Dikes::SillParameters, CartData_input)`
- `MTK_updateTracers(Grid::GridData, Arrays::NamedTuple, Tracers::StructArray, Dikes::SillParameters, time_props::TimeDependentProperties, Num::NumericalParameters)`
- `MTK_save_output(Grid::GridData, Arrays::NamedTuple, Tracers::StructArray, Dikes::SillParameters, time_props::TimeDependentProperties, Num::NumericalParameters, CartData_input::CartData)`
- `MTK_inject_dikes(Grid::GridData, Num::NumericalParameters, Arrays::NamedTuple, Mat_tup::Tuple, Dikes::SillParameters, Tracers::StructVector)`
- `MTK_initialize!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructArray, Dikes::SillParameters)`
- `MTK_finalize!(Arrays::NamedTuple, Grid::GridData, Num::NumericalParameters, Tracers::StructArray, Dikes::SillParameters, CartData_input::CartData)`

"""
@views function MTK_GeoParams(Mat_tup::Tuple, Num::NumericalParameters, Dikes::SillParameters; CartData_input::Union{Nothing, CartData} = nothing, time_props::TimeDependentProperties = TimeDepProps())

    # Change parameters based on CartData input
    if isnothing(CartData_input)
        Num.dim = Num.Ny > 0 ? 3 : 2
    else
        Num.dim = size(CartData_input.x)[3] == 1 ? 2 : 3
        if Num.dim == 2 && !hasfield(typeof(CartData_input.fields), :FlatCrossSection)
            error("You should add a Field :FlatCrossSection to your data structure with Data_Cross = addfield(Data_Cross,\"FlatCrossSection\", flatten_cross_section(Data_Cross))")
        end
        Num = MTK_GMG.Setup_Model_CartData(CartData_input, Num, Mat_tup)
    end
    Num.axisymmetric && Num.dim == 3 && error("an axisymmetric model must be 2D (Num.Ny = 0)")

    # Array & grid initializations ---------------
    Arrays = MTK_GMG.MTK_initialize_arrays(Num)

    # Set up model geometry & initial T structure
    if !isnothing(CartData_input)
        Grid = CreateGrid(CartData_input)
    elseif Num.dim == 2
        Grid = CreateGrid(size = (Num.Nx, Num.Nz), extent = (Num.W, Num.H))
    else
        Grid = CreateGrid(size = (Num.Nx, Num.Ny, Num.Nz), x = (-Num.W / 2, Num.W / 2), y = (-Num.L / 2, Num.L / 2), z = (-Num.H, 0.0))
    end
    if Num.dim == 2
        GridArray!(Arrays.R, Arrays.Z, Grid)
    else
        GridArray!(Arrays.X, Arrays.Y, Arrays.Z, Grid)
    end
    # --------------------------------------------

    Tracers = StructArray{Tracer{Num.TracerFloatType}}(undef, 1)   # Initialize tracers

    # Host buffers for advection & phases --------
    Tnew_cpu = Array{eltype(Arrays.T)}(undef, size(Arrays.T))
    Phi_melt_cpu = similar(Tnew_cpu)
    Phases = KernelAbstractions.ones(Num.backend, Int64, size(Arrays.T)...)
    Phases_init = KernelAbstractions.ones(Num.backend, Int64, size(Arrays.T)...)
    Arrays = (Arrays..., Phases = Phases, Phases_init = Phases_init)

    # Initialize Geotherm and Phases -------------
    if isnothing(CartData_input)
        MTK_GMG.MTK_initialize!(Arrays, Grid, Num, Tracers, Dikes)
    else
        MTK_GMG.MTK_initialize!(Arrays, Grid, Num, Tracers, Dikes, CartData_input)
    end
    # --------------------------------------------

    # check errors
    unique_Phases = unique(Array(Arrays.Phases))
    phase_specified = [mm.Phase for mm in Mat_tup]
    for u in unique_Phases
        if !(u in phase_specified)
            error("Properties for Phase $u are not specified in Mat_tup. Please add that")
        end
    end

    if any(isnan, Arrays.T)
        error("NaNs in T; something is wrong")
    end

    # Optionally set initial sill in models ------
    if hasproperty(Dikes, :sill) && !isnothing(Dikes.sill) && Dikes.sill isa InjectSills.CylindricalDikeTopAccretion
        c = [Dikes.sill.Center[i].val for i in 1:Num.dim]
        # CylindricalDikeTopAccretion stores the full width in W; its axis is vertical through the center
        if Num.dim == 2
            R_center = Array(Arrays.R)
        else
            R_center = sqrt.((Array(Arrays.X) .- c[1]) .^ 2 .+ (Array(Arrays.Y) .- c[2]) .^ 2)
        end
        T_init = Array(Arrays.T_init)
        T_init[(R_center .<= Dikes.sill.W.val / 2) .& (abs.(Array(Arrays.Z) .- c[end]) .< Dikes.sill.H.val / 2)] .= Dikes.T_in_Celsius
        copyto!(Arrays.T_init, T_init)
        if Num.advect_polygon == true
            if hasproperty(Dikes, :sill_poly)
                Dikes.sill_poly = InjectSills.dike_polygon(Dikes.sill)
            else
                Dikes.dike_poly = InjectSills.dike_polygon(Dikes.sill)
            end
        end
    end
    # --------------------------------------------

    # Initialize arrays --------------------------
    Arrays.Tnew .= Arrays.T_init
    Arrays.T .= Arrays.T_init

    if isdir(Num.SimName) == false
        mkdir(Num.SimName)          # create simulation directory if needed
    end
    # --------------------------------------------

    for Num.it in 1:Num.nt   # Time loop
        Num.time += Num.dt                                      # Keep track of evolved time

        # Add new dike every X years -----------------
        Tracers = MTK_GMG.MTK_inject_dikes(Grid, Num, Arrays, Mat_tup, Dikes, Tracers)
        # --------------------------------------------

        # Do a diffusion step, while taking T-dependencies into account
        Nonlinear_Diffusion_step!(Arrays, Mat_tup, Arrays.Phases, Grid, Num.dt, Num)
        # --------------------------------------------

        # Update variables ---------------------------
        # Copy fields only when tracers are active.
        if isassigned(Tracers, 1)
            copyto!(Tnew_cpu, Arrays.Tnew)
            copyto!(Phi_melt_cpu, Arrays.ϕ)

            UpdateTracers_T_ϕ!(Tracers, Grid.coord1D, Tnew_cpu, Phi_melt_cpu)      # Update info on tracers
        end

        Arrays.T .= Arrays.Tnew
        # --------------------------------------------

        # Update info on tracers ---------------------
        Tracers = MTK_GMG.MTK_updateTracers(Grid, Arrays, Tracers, Dikes, time_props, Num)
        # --------------------------------------------

        # Update time-dependent properties -----------
        MTK_GMG.MTK_update_TimeDepProps!(time_props, Grid, Num, Arrays, Mat_tup, Dikes)
        # --------------------------------------------

        # Visualize results --------------------------
        MTK_GMG.MTK_visualize_output(Grid, Num, Arrays, Mat_tup, Dikes)
        # --------------------------------------------

        # Save output to disk once in a while --------
        MTK_GMG.MTK_save_output(Grid, Arrays, Tracers, Dikes, time_props, Num, CartData_input)
        # --------------------------------------------

        # Optionally update arrays and structs (such as T or Dike) -------
        MTK_GMG.MTK_update_ArraysStructs!(Arrays, Grid, Dikes, Num, Mat_tup)
        # --------------------------------------------

        # Display output -----------------------------
        MTK_GMG.MTK_print_output(Grid, Num, Arrays, Mat_tup, Dikes)
        # --------------------------------------------

    end

    # Finalize simulation ------------------------
    MTK_GMG.MTK_finalize!(Arrays, Grid, Num, Tracers, Dikes, CartData_input)
    # --------------------------------------------

    return Grid, Arrays, Tracers, Dikes, time_props
end # end of main function
