using Tensors

abstract type AbstractMaterial end

# ── LinearElastic ────────────────────────────────────────────────────────────

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
function contravariant_elasticity(mat::LinearElastic, A_metric::SymmetricTensor{2,2,T}) where T
    Aup = inv(A_metric)
    μ = mat.E * mat.thickness / (2*(1 + mat.ν))
    λ = mat.ν * mat.thickness * mat.E / (1 - mat.ν^2)
    SymmetricTensor{4,2,T}((α,β,γ,δ) -> λ*Aup[α,β]*Aup[γ,δ] + μ*(Aup[α,γ]*Aup[β,δ] + Aup[α,δ]*Aup[β,γ]))
end

# Bending stiffness D^{αβγδ} = (t²/12) C^{αβγδ}
function contravariant_bending_stiffness(mat::LinearElastic, A_metric::SymmetricTensor{2,2,T}) where T
    (mat.thickness^2 / 12) * contravariant_elasticity(mat, A_metric)
end

# membrane_stress_and_tangent(mat, c_ms, A_metric)
#   Returns (N, C): membrane stress resultant N^{αβ} and consistent tangent C^{αβγδ}
#   evaluated at the current midsurface metric c_ms.
#
# bending_and_shear_stiffness(mat, c_ms, A_metric)
#   Returns (D, Cs): bending stiffness D^{αβγδ} and transverse shear stiffness matrix
#   Cs^{αβ} (2×2 SymmetricTensor, replaces κ_s·G·t·A^{αβ}), both at current state.
#
# Both functions have default implementations for LinearElastic that reproduce the
# existing closed-form expressions.  HyperelasticShell overrides them with derivatives
# of W evaluated at c_ms.

function membrane_stress_and_tangent(mat::LinearElastic, c_ms::SymmetricTensor{2,2,T}, A_metric) where T
    C = contravariant_elasticity(mat, A_metric)
    E = (c_ms - A_metric) / 2
    return C ⊡ E, C
end

function bending_and_shear_stiffness(mat::LinearElastic, c_ms, A_metric::SymmetricTensor{2,2,T}) where T
    D   = contravariant_bending_stiffness(mat, A_metric)
    cs  = T(5//6) * mat.E / (2*(1 + mat.ν)) * mat.thickness
    Aup = inv(A_metric)
    Cs  = SymmetricTensor{2,2,T}((cs*Aup[1,1], cs*Aup[1,2], cs*Aup[2,2]))
    return D, Cs
end

"""
    HyperelasticShell(W, thickness=1.0)

Incompressible hyperelastic shell material defined by a full 3D strain energy density
`W(C::SymmetricTensor{2,3,T}) -> T`.

The plane-stress + incompressibility constraint `det(C) = 1` is enforced internally.
`C₃₃` is determined analytically from the in-plane metric `C_αβ` and transverse shear
`C_α3 = γ_α` (no iteration):

```math
C_{33} = \\frac{1 - 2C_{12}\\gamma_1\\gamma_2 + C_{22}\\gamma_1^2 + C_{11}\\gamma_2^2}{\\det_2(C_{\\alpha\\beta})}
```

`W` can be any standard incompressible strain energy expressed in terms of the invariants
`I₁ = tr(C)`, `I₂ = ½((tr C)² − tr C²)` with `det(C) = 1`.

Example — Neo-Hookean incompressible

```julia
μ = 80.0e3; t = 1.0e-3
W_NH(C) = μ/2 * (tr(C) - 3)
mat = HyperelasticShell(W_NH, t)
```

Example — Mooney–Rivlin

```julia
c₁ = 40.0e3; c₂ = 20.0e3; t = 1.0e-3
W_MR(C) = c₁*(tr(C) - 3) + c₂*((tr(C)^2 - C ⊡ C)/2 - 3)
mat = HyperelasticShell(W_MR, t)
```
"""
struct HyperelasticShell{F, T<:AbstractFloat} <: AbstractMaterial
    W         :: F
    thickness :: T
    function HyperelasticShell(W::F, thickness::T=one(Float64)) where {F, T<:AbstractFloat}
        @assert thickness > 0 "Thickness must be positive"
        new{F, T}(W, thickness)
    end
end

# C₃₃ from det(C)=1, exact closed-form (Schur complement of the 3×3 determinant)., reduces to 1/det₂(c) when γ=0 (KL limit).
@inline get_C33(c::SymmetricTensor{2,2}, γ₁, γ₂) = (1 - 2*c[1,2]*γ₁*γ₂ + c[2,2]*γ₁^2 + c[1,1]*γ₂^2) / det(c)

# Build the full 3×3 right Cauchy–Green tensor.
# SymmetricTensor{2,3} lower-triangle column-major storage: (C₁₁,C₁₂,C₁₃,C₂₂,C₂₃,C₃₃)
@inline function build_C3D(c::SymmetricTensor{2,2}, γ₁, γ₂, C33)
    TT = promote_type(eltype(c), typeof(C33))
    SymmetricTensor{2,3,TT}((TT(c[1,1]), TT(c[1,2]), TT(γ₁), TT(c[2,2]), TT(γ₂), TT(C33)))
end

# Membrane-only reduced energy: W evaluated with γ=0 and C₃₃=1/det(c).
# Used by membrane_stress_and_tangent via Tensors.hessian — the C₃₃ substitution is
# differentiated through automatically, giving the condensed plane-stress tangent.
@inline function W_membrane(mat::HyperelasticShell, c::SymmetricTensor{2,2,T}) where T
    C33 = 1 / det(c)
    mat.W(build_C3D(c, zero(T), zero(T), C33))
end

# Membrane stress and consistent tangent via a single nested gradient pass on the
# 3-component function W_membrane.  Cost: O(9) W evaluations per QP — much cheaper
# than differentiating through the full 5n-DOF energy with ForwardDiff.
function membrane_stress_and_tangent(mat::HyperelasticShell, c_ms::SymmetricTensor{2,2}, A_metric)
    ∇W(c) = gradient(x -> W_membrane(mat, x), c)
    H, S  = gradient(∇W, c_ms, :all)   # H = ∂²W_mem/∂c², S = ∂W_mem/∂c at c_ms
    N = 2 * mat.thickness * S           # N^{αβ} = 2t ∂W/∂C_{αβ}; factor 2 because
    C = 4 * mat.thickness * H           # Tensors.jl stores off-diagonal as ½·∂W/∂c₁₂
    return N, C
end

# Bending stiffness: (t²/12)·C from the same membrane tangent.
# Shear stiffness: 2×2 hessian of W w.r.t. γ at γ=0, multiplied by t.
# This replaces the scalar κ_s·G·t·A^{αβ} with the exact linearised shear stiffness
# derived from W, without a separate shear modulus parameter.
function bending_and_shear_stiffness(mat::HyperelasticShell, c_ms::SymmetricTensor{2,2,T}, A_metric) where T
    _, C = membrane_stress_and_tangent(mat, c_ms, A_metric)
    D    = (mat.thickness^2 / 12) * C
    W_sh(γ) = let C33 = get_C33(c_ms, γ[1], γ[2])
        mat.W(build_C3D(c_ms, γ[1], γ[2], C33))
    end
    Cs_full = mat.thickness * hessian(W_sh, zero(Vec{2,T}))
    Cs = SymmetricTensor{2,2,T}((Cs_full[1,1], Cs_full[1,2], Cs_full[2,2]))
    return D, Cs
end

# contravariant_elasticity / contravariant_bending_stiffness kept for KL assembly
@inline contravariant_elasticity(mat::HyperelasticShell, c::SymmetricTensor{2,2}) = 4 * mat.thickness * hessian(x -> W_membrane(mat, x), c)
@inline contravariant_bending_stiffness(mat::HyperelasticShell, c::SymmetricTensor{2,2}) = (mat.thickness^2 / 12) * contravariant_elasticity(mat, c)
