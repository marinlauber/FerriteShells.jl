using Tensors

abstract type AbstractMaterial end

struct LinearElastic{T} <: AbstractMaterial
    E::T
    ν::T
    thickness::T
    H :: SymmetricTensor{4,2,T}
    C :: SymmetricTensor{4,2,T}
    function LinearElastic(E::T, ν, thickness) where T
        @assert E > 0 "Young's modulus must be positive"
        @assert 0 ≤ ν < 0.5 "Poisson's ratio must be in [0, 0.5)"
        @assert thickness > 0 "Thickness must be positive"
        factor = E / (1 - ν^2)
        H = zeros(2,2,2,2)
        H[1,1,1,1] = thickness * factor
        H[1,1,2,2] = thickness * factor * ν
        H[2,2,1,1] = thickness * factor * ν
        H[2,2,2,2] = thickness * factor
        # TODO not sure how, but the N_12 = t * (factor*(1-ν)/2 * 2E_12) term is in engineering strain
        H[1,2,1,2] = thickness * factor * (1 - ν) / 2
        H[2,1,2,1] = thickness * factor * (1 - ν) / 2
        H[1,2,2,1] = thickness * factor * (1 - ν) / 2
        H[2,1,1,2] = thickness * factor * (1 - ν) / 2
        C = zeros(2,2,2,2)
        # Populate using minor symmetries
        C[1,1,1,1] = thickness * factor
        C[2,2,2,2] = thickness * factor
        C[1,1,2,2] = thickness * factor * ν
        C[2,2,1,1] = thickness * factor * ν
        C[1,2,1,2] = thickness * factor * (1 - ν) / 2
        C[2,1,2,1] = thickness * factor * (1 - ν) / 2
        C[1,2,2,1] = thickness * factor * (1 - ν) / 2
        C[2,1,1,2] = thickness * factor * (1 - ν) / 2
        new{typeof(E)}(E, ν, thickness, SymmetricTensor{4,2,T}(H), SymmetricTensor{4,2,T}(C))
    end
end

# Contravariant elasticity tensor C^{αβγδ} = λ A^{αβ}A^{γδ} + μ(A^{αγ}A^{βδ} + A^{αδ}A^{βγ})
# where A^{αβ} = inv(A_{αβ}) is the contravariant reference metric.
# For a unit-square element A^{αβ} = δ^{αβ} and this reduces to mat.C.
function contravariant_elasticity(mat::LinearElastic, A_metric::SymmetricTensor{2,2,T}) where T
    det_A = A_metric[1,1]*A_metric[2,2] - A_metric[1,2]^2
    A11u  =  A_metric[2,2] / det_A
    A12u  = -A_metric[1,2] / det_A
    A22u  =  A_metric[1,1] / det_A
    Aup   = SymmetricTensor{2,2,T}((A11u, A12u, A22u))

    factor = mat.E / (1 - mat.ν^2)
    μ = mat.E * mat.thickness / (2*(1 + mat.ν))
    λ = mat.ν * mat.thickness * factor

    C = zeros(T, 2, 2, 2, 2)
    for α in 1:2, β in 1:2, γ in 1:2, δ in 1:2
        C[α,β,γ,δ] = λ*Aup[α,β]*Aup[γ,δ] + μ*(Aup[α,γ]*Aup[β,δ] + Aup[α,δ]*Aup[β,γ])
    end
    return SymmetricTensor{4,2,T}(C)
end

"""
    membrane_stress(material, E::SymmetricTensor{2,2,T}) -> N

Compute membrane stress for linear isotropic membrane.
"""
function membrane_stress(material::LinearElastic{T},
                         E::SymmetricTensor{2,2,T}) where T
    E_mod, ν, t = material.E, material.ν, material.thickness
    factor = E_mod / (1 - ν^2)

    # directly compute stress
    N11 = t * (factor*E[1,1] + factor*ν*E[2,2])
    N22 = t * (factor*ν*E[1,1] + factor*E[2,2])
    N12 = t * (factor*(1-ν)/2 * 2E[1,2])
    return SymmetricTensor{2,2,T}((N11,N12,N22))
end

"""
    membrane_tangent(material) -> C

Compute 4th-order tangent for linear isotropic membrane tensor
"""
function membrane_tangent(material::LinearElastic{T}) where T
    E_mod, ν, t = material.E, material.ν, material.thickness
    factor = E_mod / (1 - ν^2)
    # 4th order tensor
    C = zeros(2,2,2,2)
    # Populate using minor symmetries
    C[1,1,1,1] = t * factor
    C[2,2,2,2] = t * factor
    C[1,1,2,2] = t * factor * ν
    C[2,2,1,1] = t * factor * ν
    C[1,2,1,2] = t * factor * (1 - ν) / 2
    C[2,1,2,1] = t * factor * (1 - ν) / 2
    C[1,2,2,1] = t * factor * (1 - ν) / 2
    C[2,1,1,2] = t * factor * (1 - ν) / 2
    return SymmetricTensor{4,2}(C)
end