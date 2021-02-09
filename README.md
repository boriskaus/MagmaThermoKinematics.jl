# ZirconThermoKinematics.jl

This package can be used to simulate the thermal evolution of magmatic systems, consisting of (kinematically) emplaced dikes. 
It can simulate 2D, 2D axisymmetric and 3D systems, and works in parallel on both CPU (and GPU's). We use a finite difference discretization, combined with a semi-lagrangian temperature advection solver and tracers to track the thermal evolution particles. Dikes are emplaced kinematically and the host rock is shifted to make space; cooling and crystallizing of melt is taken into account, and we provide.
