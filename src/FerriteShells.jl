module FerriteShells

using Reexport
@reexport using Ferrite
@reexport using Tensors

using Base: @propagate_inbounds

import Ferrite: reinit!

include("shellcellvalues.jl")
export ShellCellValues

include("kinematics.jl")
export kinematics

include("material.jl")
export LinearElastic, contravariant_elasticity

include("assembly.jl")
export membrane_residuals_KL!, membrane_tangent_KL!, bending_residuals_KL!, bending_tangent_KL!
export membrane_residuals_RM!, membrane_tangent_RM!, bending_residuals_RM!, bending_tangent_RM!

include("utils.jl")
export shell_grid, assemble_traction!, assemble_pressure!, assemble_pressure_tangent!, shelldofs

end # module FerriteShells
