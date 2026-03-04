using Tensors

struct LinearMembraneMaterial{T}
    E::T
    ν::T
    thickness::T
end

function membrane_stress(material::LinearMembraneMaterial{T}, E::SymmetricTensor{2,2,T}) where T

    E_mod = material.E
    ν = material.ν
    t = material.thickness

    factor = E_mod / (1 - ν^2)

    C11 = factor
    C12 = factor * ν
    C33 = factor * (1 - ν) / 2

    # Voigt components
    E11 = E[1,1]
    E22 = E[2,2]
    E12 = E[1,2]

    N11 = t * (C11*E11 + C12*E22)
    N22 = t * (C12*E11 + C11*E22)
    N12 = t * (C33*2E12)

    return SymmetricTensor{2,2,T}(N11, N12, N22)
end

"""
    membrane_stress_and_tangent(material, E::SymmetricTensor{2,2,T}) -> (N, C)

Compute membrane stress and 4th-order tangent for linear isotropic membrane.
"""
function membrane_stress_and_tangent(material::LinearMembraneMaterial{T},
                                     E::SymmetricTensor{2,2,T}) where T
    # Material parameters
    E_mod = material.E
    ν     = material.ν
    t     = material.thickness

    factor = E_mod / (1 - ν^2)

    # 2D plane stress C tensor
    # C_{11,11} = C11 = factor
    # C_{11,22} = C12 = factor * ν
    # C_{22,22} = C11
    # C_{12,12} = C33 = factor * (1-ν)/2

    C = zeros(SymmetricTensor{2,2,SymmetricTensor{2,2,T}})

    # Voigt-style mapping for clarity:
    # indices (αβ,γδ): 11→1, 22→2, 12→3
    # But we can write in tensor contraction form

    C[1,1] = factor     # C11
    C[1,2] = factor*ν   # C12
    C[2,1] = factor*ν   # C21
    C[2,2] = factor     # C22
    C[1,3] = 0
    C[2,3] = 0
    C[3,1] = 0
    C[3,2] = 0
    C[3,3] = factor*(1-ν)/2  # shear

    # Compute membrane stress
    E11 = E[1,1]
    E22 = E[2,2]
    E12 = E[1,2]

    N11 = t*(C[1,1]*E11 + C[1,2]*E22)
    N22 = t*(C[2,1]*E11 + C[2,2]*E22)
    N12 = t*(C[3,3]*2*E12)  # engineering shear: factor 2

    N = SymmetricTensor{2,2,T}(N11, N12, N22)

    # 4th-order tangent in SymmetricTensor{2,2} space
    # Only small 3x3 representation needed for explicit contraction
    # For explicit residual/tangent, we can pass this C along
    return N, C
end