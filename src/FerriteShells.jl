module FerriteShells

using Reexport
@reexport using Ferrite
@reexport using Tensors

using Base: @propagate_inbounds

import Ferrite: reinit!

include("mitc.jl")
export AbstractMITC, NoMITC, MITC

include("shellcellvalues.jl")
export ShellCellValues

include("material.jl")
export LinearElastic, Hyperelastic
export membrane_stress_and_tangent, bending_and_shear_stiffness

include("kinematics.jl")

include("membrane.jl")
export membrane_residuals!, membrane_tangent!

include("kirchhofflove.jl")
export bending_residuals_KL!, bending_tangent_KL!

include("reissnermindlin.jl")
export bending_residuals_RM!, bending_tangent_RM!
export residuals_RM!, tangent_RM!, residuals_and_tangent_RM!
export residuals_RM_FD!, tangent_RM_FD!

include("assembly.jl")
export assemble_pressure!, assemble_pressure_tangent!, assemble_traction!, apply_pointload!, mass_matrix!

include("utils.jl")
export shell_grid, get_ferrite_grid, compute_volume, volume_residual, volume_gradient!, director_field
export shell_strains, embed23

include("nodeframe.jl")
export NodeFrames, add_director_symmetry!

end # module FerriteShells
