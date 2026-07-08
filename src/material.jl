using Tensors

abstract type AbstractMaterial end

"""
    LinearElastic(E, ν, thickness=1.0; β=1.0)

Linear elastic shell material defined by Young's modulus `E`, Poisson's ratio `ν`,
and thickness `thickness`.

`β` is a dimensionless **bending scale factor** that decouples the bending/shear
response from the membrane response: the bending stiffness becomes
`D = β · C · t³/12` and the transverse-shear stiffness scales likewise, while the
membrane stiffness `A = C · t` is unaffected. `β = 1` recovers the physical shell;
`β → 0` approaches a pure membrane (tension-field limit). At exactly `β = 0` the
rotation DOFs are unconstrained and the assembled tangent is singular — use a small
positive `β` (e.g. `1e-4`) for a near-membrane response.
"""
struct LinearElastic{T} <: AbstractMaterial
    E::T
    ν::T
    thickness::T
    β::T
    function LinearElastic(E::T, ν::T, thickness::T=one(T); β::T=one(T)) where T
        @assert E > 0 "Young's modulus must be positive"
        @assert 0 ≤ ν < 0.5 "Poisson's ratio must be in [0, 0.5)"
        @assert thickness > 0 "Thickness must be positive"
        @assert β ≥ 0 "Bending scale factor must be non-negative"
        new{typeof(E)}(E, ν, thickness, β)
    end
end

# LinearElastic: frame arguments accepted but ignored.
function membrane_stress_and_tangent(mat::LinearElastic, c_ms::SymmetricTensor{2,2,T},
                                     A_metric, A₁=nothing, A₂=nothing, G₃=nothing) where T
    Aup = inv(A_metric)
    μ = mat.E * mat.thickness / (2*(1 + mat.ν))
    λ = mat.ν * mat.thickness * mat.E / (1 - mat.ν^2)
    # C^{αβγδ} = λ Aup^{αβ} Aup^{γδ} + μ (Aup^{αγ} Aup^{βδ} + Aup^{αδ} Aup^{βγ})
    C = λ * (Aup ⊗ Aup) + μ * symmetric(otimesu(Aup, Aup) + otimesl(Aup, Aup))
    return C ⊡ ((c_ms - A_metric) / 2), C
end

# LinearElastic: frame arguments accepted but ignored.
function bending_and_shear_stiffness(mat::LinearElastic, c_ms,
                                     A_metric::SymmetricTensor{2,2,T},
                                     A₁=nothing, A₂=nothing, G₃=nothing) where T
    _, C = membrane_stress_and_tangent(mat, c_ms, A_metric)
    D    = mat.β * (mat.thickness^2 / 12) * C
    cs   = mat.β * T(5//6) * mat.E / (2*(1 + mat.ν)) * mat.thickness
    Aup  = inv(A_metric)
    Cs   = SymmetricTensor{2,2,T}((cs*Aup[1,1], cs*Aup[1,2], cs*Aup[2,2]))
    return D, Cs
end

"""
    Hyperelastic(W, thickness=1.0)

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
mat = Hyperelastic(W_NH, t)
```

Example — Mooney–Rivlin

```julia
c₁ = 40.0e3; c₂ = 20.0e3; t = 1.0e-3
W_MR(C) = c₁*(tr(C) - 3) + c₂*((tr(C)^2 - C ⊡ C)/2 - 3)
mat = Hyperelastic(W_MR, t)
```
"""
struct Hyperelastic{F, T<:AbstractFloat} <: AbstractMaterial
    W         :: F
    thickness :: T
    function Hyperelastic(W::F, thickness::T=one(Float64)) where {F, T<:AbstractFloat}
        @assert thickness > 0 "Thickness must be positive"
        new{F, T}(W, thickness)
    end
end

# C₃₃ from det(C_nat) = det_A so that det(C_cart) = 1 (physical incompressibility).
# det_A = det(A_metric) = |A₁ × A₂|² (reference area element squared).
# Reduces to det_A/det₂(c) when γ=0 (KL / no-shear limit).
@inline get_C33(c::SymmetricTensor{2,2}, γ₁, γ₂, det_A) = (det_A + c[2,2]*γ₁^2 - 2*c[1,2]*γ₁*γ₂ + c[1,1]*γ₂^2) / det(c)

# Build the full 3×3 right Cauchy–Green tensor.
# SymmetricTensor{2,3} lower-triangle column-major storage: (C₁₁,C₁₂,C₁₃,C₂₂,C₂₃,C₃₃)
@inline function build_C3D(c::SymmetricTensor{2,2}, γ₁, γ₂, C33)
    TT = promote_type(eltype(c), typeof(C33))
    SymmetricTensor{2,3,TT}((TT(c[1,1]), TT(c[1,2]), TT(γ₁), TT(c[2,2]), TT(γ₂), TT(C33)))
end

# Reference Jacobian: columns = A₁, A₂, G₃ in Cartesian.  Stored column-major.
@inline _J_ref(A₁, A₂, G₃) = Tensor{2,3}((A₁[1],A₁[2],A₁[3], A₂[1],A₂[2],A₂[3], G₃[1],G₃[2],G₃[3]))

# Transform C_nat (natural frame) → C_cart (Cartesian): C_cart = Jinv' C_nat Jinv.
@inline _to_C_cart(C_nat::SymmetricTensor{2,3}, Jinv::Tensor{2,3}) = symmetric(Jinv' ⋅ Tensor{2,3}(C_nat) ⋅ Jinv)

# Evaluate W at the physical Cartesian C, no shear (γ=0).
@inline function _W_phys(mat::Hyperelastic, c::SymmetricTensor{2,2}, det_A, Jinv)
    C33 = det_A / det(c)
    mat.W(_to_C_cart(build_C3D(c, zero(eltype(c)), zero(eltype(c)), C33), Jinv))
end

# Evaluate W at the physical Cartesian C, with shear γ₁, γ₂.
@inline function _W_phys(mat::Hyperelastic, c::SymmetricTensor{2,2}, γ₁, γ₂, det_A, Jinv)
    C33 = get_C33(c, γ₁, γ₂, det_A)
    mat.W(_to_C_cart(build_C3D(c, γ₁, γ₂, C33), Jinv))
end

# Membrane stress N and consistent tangent C via nested gradient of _W_phys.
# N^{αβ} = 2t ∂W/∂C_{αβ}; factor 2 from Tensors.jl Mandel off-diagonal convention.
function membrane_stress_and_tangent(mat::Hyperelastic, c_ms::SymmetricTensor{2,2},
                                     A_metric, A₁, A₂, G₃)
    det_A = det(A_metric)
    Jinv  = inv(_J_ref(A₁, A₂, G₃))
    ∇W(c) = gradient(x -> _W_phys(mat, x, det_A, Jinv), c)
    H, S  = gradient(∇W, c_ms, :all)
    N = 2 * mat.thickness * S
    C = 4 * mat.thickness * H
    return N, C
end

# Bending and shear stiffness tensors in the physical Cartesian frame.
function bending_and_shear_stiffness(mat::Hyperelastic, c_ms::SymmetricTensor{2,2,T},
                                     A_metric, A₁, A₂, G₃) where T
    _, C  = membrane_stress_and_tangent(mat, c_ms, A_metric, A₁, A₂, G₃)
    D     = (mat.thickness^2 / 12) * C
    det_A = det(A_metric)
    Jinv  = inv(_J_ref(A₁, A₂, G₃))
    W_sh(γ) = _W_phys(mat, c_ms, γ[1], γ[2], det_A, Jinv)
    Cs_full = mat.thickness * hessian(W_sh, zero(Vec{2,T}))
    Cs = SymmetricTensor{2,2,T}((Cs_full[1,1], Cs_full[1,2], Cs_full[2,2]))
    return D, Cs
end