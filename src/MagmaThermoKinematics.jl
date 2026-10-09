"""
Module MagmaThermoKinematics

Enables Earth Scientists to simulate the thermal evolution of magmatic systems.

"""
module MagmaThermoKinematics

# list required modules
using Reexport
using Random                                    # random numbers
using StructArrays                              # for tracers and dike polygon
using Parameters                                # More flexible definition of parameters
using Interpolations                            # Fast interpolations
using StaticArrays
using JLD2                                      # Load/save data to disk
@reexport using InjectSills                     # Re-export InjectSills API (sill constructors + helpers)
@reexport using GeoParams                                 # Material parameters calculations
using KernelAbstractions                        # CPU and GPU kernels; GPU backends come with CUDA.jl, Metal.jl, ...

abstract type NumericalParameters end
abstract type SillParameters end
abstract type TimeDependentProperties end

include("Units.jl")                             # various useful units

# Few useful parameters
const SecYear = 3600 * 24 * 365.25
const kyr = 1000 * SecYear
const Myr = 1.0e6 * SecYear
const km³ = 1000^3
export SecYear, kyr, Myr, km³

export NumericalParameters, SillParameters, TimeDependentProperties

include("Grid.jl")
using .Grid
export GridData, CreateGrid

# Routines that deal with tracers
include("Tracers.jl")
export UpdateTracers, AdvectTracers!, InitializeTracers, PhaseRatioFromTracers, CorrectTracersForTopography!
export RockAssemblage, update_Tvec!
export PhaseRatioFromTracers!, PhasesFromTracers!, UpdateTracers_T_ϕ!, UpdateTracers_Field! # new routines
export Tracer, TracersToGrid!

include("MeltingRelationships.jl")
export SolidFraction, ComputeLithostaticPressure, LoadPhaseDiagrams, PhaseDiagramData, ComputeDensityAndPressure
export PhaseRatioAverage!, ComputeSeismicVelocities, SolidFraction_Parameterized!

# Export functions that will be available outside this module
export StructArray, LazyRow # useful
export Tracer

include("InjectSills_utils.jl")
export inject_sills, add_dike

# routines related to advection & interpolation
include("Advection.jl")
export AdvectTemperature, AdvectTemperature!, Interpolate!, CorrectBounds, evaluate_interp_2D, evaluate_interp_3D

include("Utils.jl")
export Process_ZirconAges, simulate_zircon_growth_from_tracers, volume_averaged_age, copy_arrays_GPU2CPU!, copy_arrays_CPU2GPU!

include("MTK_GMG_structs.jl")
export NumParam, SillParams, TimeDepProps

include("Fields.jl")
export CreateArrays

include("Diffusion.jl")
export Numeric_params, Nonlinear_Diffusion_step!, diffusion_step!, compute_phase_param!, GridArray!, bc_zero_flux!, bc_T!, bc_z_bottom_flux!

include("MTK_GMG.jl")

include("solver.jl")
export MTK_GeoParams

# KernelAbstractions' CPU backend; GPU backends are exported by their own packages
export CPU

# Routines related to Parameters.jl, which come in handy in the main routine
export @unpack, @with_kw


end # module
