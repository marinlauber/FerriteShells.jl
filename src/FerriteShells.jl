module FerriteShells

using Reexport
@reexport using Ferrite
@reexport using Tensors

using Base: @propagate_inbounds

import Ferrite: reinit!

abstract type AbstractStrainMeasure end
struct LinearStrain <: AbstractStrainMeasure end
struct GreenLagrangeStrain <: AbstractStrainMeasure end
export LinearStrain, GreenLagrangeStrain

include("mitc.jl")
export AbstractMITC, NoMITC, MITC, MITC9

include("shellcellvalues.jl")
export ShellCellValues

include("kinematics.jl")
export kinematics, kinematics_strains

include("material.jl")
export LinearElastic, contravariant_elasticity

include("assembly.jl")
export membrane_residuals_KL!, membrane_tangent_KL!, bending_residuals_KL!, bending_tangent_KL!
export membrane_residuals_RM!, membrane_tangent_RM!, bending_residuals_RM!, bending_tangent_RM!
export membrane_residuals_RM_FD!, membrane_tangent_RM_FD!, bending_residuals_RM_FD!, bending_tangent_RM_FD!
export assemble_pressure!, assemble_pressure_tangent!, assemble_traction!, apply_pointload!, mass_matrix!

include("utils.jl")
export shell_grid, shelldofs, get_ferrite_grid, compute_volume, volume_residual, volume_gradient!, director_field

end # module FerriteShells
