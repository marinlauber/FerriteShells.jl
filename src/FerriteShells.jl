module FerriteShells

using Reexport
@reexport using Ferrite
@reexport using Tensors

using Base: @propagate_inbounds

import Ferrite: reinit!

include("mitc.jl")
export AbstractMITC, NoMITC, MITC, MITC9

include("shellcellvalues.jl")
export ShellCellValues

include("material.jl")
export LinearElastic, Hyperelastic
export membrane_stress_and_tangent, bending_and_shear_stiffness

include("assembly.jl")
export membrane_residuals_KL!, membrane_tangent_KL!, bending_residuals_KL!, bending_tangent_KL!
export membrane_residuals_RM!, membrane_tangent_RM!, bending_residuals_RM!, bending_tangent_RM!
export residuals_RM_FD!, tangent_RM_FD!
export assemble_pressure!, assemble_pressure_tangent!, assemble_traction!, apply_pointload!, mass_matrix!

include("utils.jl")
export shell_grid, shelldofs, shelldofs!, get_ferrite_grid, compute_volume, volume_residual, volume_gradient!, director_field
export shell_strains, embed23, NodeFrames

end # module FerriteShells
