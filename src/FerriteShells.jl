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
export membrane_residuals!, membrane_tangent!

include("utils.jl")
export shell_grid, assemble_traction!, assemble_pressure!, assemble_pressure_tangent!

end # module FerriteShells
