using Tensors

abstract type AbstractMaterial end

struct LinearElastic{T} <: AbstractMaterial
    E::T
    ν::T
    thickness::T
    function LinearElastic(E::T, ν, thickness=one(T)) where T
        @assert E > 0 "Young's modulus must be positive"
        @assert 0 ≤ ν < 0.5 "Poisson's ratio must be in [0, 0.5)"
        @assert thickness > 0 "Thickness must be positive"
        new{typeof(E)}(E, ν, thickness)
    end
end

# Contravariant elasticity tensor C^{αβγδ} = λ A^{αβ}A^{γδ} + μ(A^{αγ}A^{βδ} + A^{αδ}A^{βγ})
# where A^{αβ} = inv(A_{αβ}) is the contravariant reference metric.
# For a unit-square element A^{αβ} = δ^{αβ} and this reduces to mat.C.
function contravariant_elasticity(mat::LinearElastic, A_metric::SymmetricTensor{2,2,T}) where T
    # Compute the contravariant metric A^{αβ} = inv(A_{αβ}) from the covariant metric A_{αβ}.
    Aup = inv(A_metric) # implemented in Tensors.jl

    # Lame parameters scaled by thickness (for plane stress)
    μ = mat.E * mat.thickness / (2*(1 + mat.ν))
    λ = mat.ν * mat.thickness * mat.E / (1 - mat.ν^2)

    # use implicit function constructor to build the 4th-order elasticity tensor from the contravariant metric
    SymmetricTensor{4,2,T}((α,β,γ,δ) -> λ*Aup[α,β]*Aup[γ,δ] + μ*(Aup[α,γ]*Aup[β,δ] + Aup[α,δ]*Aup[β,γ]))
end
