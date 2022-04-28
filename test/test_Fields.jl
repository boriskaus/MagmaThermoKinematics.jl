using Test
using MagmaThermoKinematics

environment!(:cpu, Float64, 2) 

@testset "Fields" begin

Arrays = CreateArrays(Dict( (100,100)=>(A=1.1,B=1,C=1.2), 
                            (101,100)=>(E=0,))) 

@test Arrays.A[1] == 1.1
@test sum(Arrays.B) == 100*100
@test size(Arrays.E) == (101,100)



end
