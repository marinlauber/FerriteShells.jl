using Tensors
using ForwardDiff

abstract type AbstractMaterial end

"""
    LinearElastic(E, ν, thickness=1.0; tension_field=false, ε_tf=1e-3)

Linear elastic shell material defined by Young's modulus `E`, Poisson's ratio `ν`,
and thickness `thickness`.

`tension_field=true` enables a Roddeman wrinkling relaxation of the membrane stress:
a thin membrane cannot carry compression, so where the minor principal membrane
stress goes negative it is relaxed (uniaxial tension along the major axis, or slack
if both principal stresses are ≤ 0).  `ε_tf` is a small positive-definiteness floor
kept on the relaxed tangent.  Bending/shear stiffness is unaffected.
"""
struct LinearElastic{T} <: AbstractMaterial
    E::T
    ν::T
    thickness::T
    tension_field::Bool
    ε_tf::T
    function LinearElastic(E::T, ν::T, thickness::T=one(T);
                           tension_field::Bool=false, ε_tf::T=T(1e-3)) where T
        @assert E > 0 "Young's modulus must be positive"
        @assert 0 ≤ ν < 0.5 "Poisson's ratio must be in [0, 0.5)"
        @assert thickness > 0 "Thickness must be positive"
        @assert ε_tf ≥ 0 "tension-field floor must be non-negative"
        new{typeof(E)}(E, ν, thickness, tension_field, ε_tf)
    end
end

# Contravariant plane-stress elasticity C^{αβγδ} = λ Aup^{αβ} Aup^{γδ}
# + μ (Aup^{αγ} Aup^{βδ} + Aup^{αδ} Aup^{βγ}).  Un-relaxed — shared by the membrane
# stress and the bending/shear stiffness (which must NOT wrinkle-relax).
function contravariant_elasticity(mat::LinearElastic, A_metric)
    Aup = inv(A_metric)
    μ = mat.E * mat.thickness / (2*(1 + mat.ν))
    λ = mat.ν * mat.thickness * mat.E / (1 - mat.ν^2)
    λ * (Aup ⊗ Aup) + μ * symmetric(otimesu(Aup, Aup) + otimesl(Aup, Aup))
end

# Roddeman wrinkling relaxation of a contravariant membrane stress N^{αβ}.  The
# principal stresses live in an orthonormal tangent frame, so transform with the
# covariant basis (A₁, A₂), project, and transform back.  σ₂ ≥ 0: taut (unchanged);
# σ₁ > 0 > σ₂: wrinkled (keep only the major tension eigenpair); else slack (→ 0).
# The ε·Ñ floor keeps the relaxed tangent positive-definite.
function tension_field_relax(N::SymmetricTensor{2,2}, A₁, A₂, ε)
    e₁ = A₁ / norm(A₁)
    e₂ = A₂ - (A₂ ⋅ e₁) * e₁;  e₂ = e₂ / norm(e₂)
    P  = Tensor{2,2}((e₁ ⋅ A₁, e₂ ⋅ A₁, e₁ ⋅ A₂, e₂ ⋅ A₂))   # Pᵢα = eᵢ·Aα
    Ñ  = symmetric(P ⋅ N ⋅ transpose(P))                      # physical stress
    eg = eigen(Ñ);  σ = eg.values;  V = eg.vectors            # σ[1] ≤ σ[2]
    Ñr = if σ[1] ≥ 0
        Ñ
    elseif σ[2] > 0
        nI = V ⋅ basevec(Vec{2}, 2)                           # major eigenvector
        σ[2] * symmetric(nI ⊗ nI) + ε * Ñ
    else
        ε * Ñ
    end
    Pi = inv(P)
    symmetric(Pi ⋅ Ñr ⋅ transpose(Pi))                        # back to contravariant
end

# Minor (min) principal value of a contravariant stress N^{αβ} in the orthonormal
# tangent frame — the wrinkling criterion (< 0 ⇒ relaxation active).
function _min_principal_stress(N::SymmetricTensor{2,2}, A₁, A₂)
    e₁ = A₁ / norm(A₁)
    e₂ = A₂ - (A₂ ⋅ e₁) * e₁;  e₂ = e₂ / norm(e₂)
    P  = Tensor{2,2}((e₁ ⋅ A₁, e₂ ⋅ A₁, e₁ ⋅ A₂, e₂ ⋅ A₂))
    eigvals(symmetric(P ⋅ N ⋅ transpose(P)))[1]
end

function membrane_stress_and_tangent(mat::LinearElastic, c_ms::SymmetricTensor{2,2,T},
                                     A_metric, A₁=nothing, A₂=nothing, G₃=nothing) where T
    C = contravariant_elasticity(mat, A_metric)
    N = C ⊡ ((c_ms - A_metric) / 2)
    (mat.tension_field && A₁ !== nothing) || return N, C
    # taut (both principal stresses ≥ 0): relaxation is the identity, tangent = C.
    _min_principal_stress(N, A₁, A₂) ≥ 0 && return N, C
    # wrinkled/slack: relaxed stress + consistent tangent via ForwardDiff.
    relaxed(c) = tension_field_relax(C ⊡ ((c - A_metric) / 2), A₁, A₂, mat.ε_tf)
    return relaxed(c_ms), 2 * Tensors.gradient(relaxed, c_ms)   # ∂N/∂E = 2·∂N/∂c_ms
end

# Membrane stress only (no tangent) — used by the residual assembly / line search
# so the tension-field path skips the ForwardDiff tangent.  Generic fallback keeps
# any other material correct.
membrane_stress(mat, c_ms, A_metric, A₁=nothing, A₂=nothing, G₃=nothing) =
    membrane_stress_and_tangent(mat, c_ms, A_metric, A₁, A₂, G₃)[1]
function membrane_stress(mat::LinearElastic, c_ms::SymmetricTensor{2,2}, A_metric,
                         A₁=nothing, A₂=nothing, G₃=nothing)
    N = contravariant_elasticity(mat, A_metric) ⊡ ((c_ms - A_metric) / 2)
    (mat.tension_field && A₁ !== nothing) || return N
    tension_field_relax(N, A₁, A₂, mat.ε_tf)
end

# LinearElastic: frame arguments accepted but ignored.  Uses the UN-relaxed C so
# bending/shear never inherits the membrane wrinkling relaxation.
function bending_and_shear_stiffness(mat::LinearElastic, c_ms,
                                     A_metric::SymmetricTensor{2,2,T},
                                     A₁=nothing, A₂=nothing, G₃=nothing) where T
    C    = contravariant_elasticity(mat, A_metric)
    D    = (mat.thickness^2 / 12) * C
    cs   = T(5//6) * mat.E / (2*(1 + mat.ν)) * mat.thickness
    Aup  = inv(A_metric)
    Cs   = SymmetricTensor{2,2,T}((cs*Aup[1,1], cs*Aup[1,2], cs*Aup[2,2]))
    return D, Cs
end

"""
    Hyperelastic(W, thickness=1.0; incompressible=true)

Hyperelastic shell material defined by a full 3D strain energy density
`W(C::SymmetricTensor{2,3,T}) -> T`.

The through-thickness component `C₃₃` of the natural-frame right Cauchy–Green tensor is
not a degree of freedom of the shell — it is condensed out of `W`, and `incompressible`
selects which condition does it.

`incompressible=true` (default) enforces the *kinematic* constraint `det(C) = 1`, which
gives `C₃₃` analytically from the in-plane metric `C_αβ` and transverse shear `C_α3 = γ_α`
(no iteration):

```math
C_{33} = \\frac{\\det A + C_{22}\\gamma_1^2 - 2C_{12}\\gamma_1\\gamma_2 + C_{11}\\gamma_2^2}{\\det_2(C_{\\alpha\\beta})}
```

Because the reduced energy is then differentiated with `C₃₃(C_αβ)` substituted, the chain
rule reproduces exactly the pressure multiplier that `S³³ = 0` would give, so plane stress
holds as well.  This is the correct — and cheapest — choice for a genuinely incompressible
`W` (Neo-Hookean, Mooney–Rivlin, …).

`incompressible=false` enforces the *static* plane-stress condition `S³³ = 2 ∂W/∂C₃₃ = 0`
instead, solved by Newton on `C₃₃` (started from the incompressible value, quadratic
convergence; exact in one step for energies quadratic in the Green–Lagrange strain).
Use this for any compressible `W` — Saint-Venant–Kirchhoff, compressible Neo-Hookean, or
anything carrying a volumetric term `U(J)`, all of which are otherwise silently forced to
`ν = 0.5`.  `W` must depend on `C₃₃` (every physical 3D energy does).

Transverse shear carries the same `κ_s = 5/6` correction as [`LinearElastic`](@ref),
applied as `γ → √κ_s·γ` before `W` is evaluated.

Note: `W` runs inside nested `ForwardDiff` (and, under `incompressible=false`, a
Newton iteration underneath that), so it must be type-stable — closing over
non-`const` globals allocates on every call. Prefer a callable struct, as in the
Saint-Venant–Kirchhoff example below.

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

Example — Saint-Venant–Kirchhoff (compressible, `ν = 0.3`), type-stable functor form

```julia
struct SVKEnergy{T}
    λ::T
    μ::T
end
(w::SVKEnergy)(C) = (Eg = (C - one(C))/2; w.λ/2 * tr(Eg)^2 + w.μ * (Eg ⊡ Eg))

E = 0.35e8; ν = 0.3; t = 0.2e-3
mat = Hyperelastic(SVKEnergy(E*ν/((1+ν)*(1-2ν)), E/(2*(1+ν))), t; incompressible=false)
# ≡ LinearElastic(E, ν, t)
```
"""
struct Hyperelastic{F, T<:AbstractFloat} <: AbstractMaterial
    W              :: F
    thickness      :: T
    incompressible :: Bool
    function Hyperelastic(W::F, thickness::T=one(Float64);
                          incompressible::Bool=true) where {F, T<:AbstractFloat}
        @assert thickness > 0 "Thickness must be positive"
        new{F, T}(W, thickness, incompressible)
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

# Strip every layer of ForwardDiff nesting — used to branch on primal values only.
@inline _rawvalue(x::Real) = x
@inline _rawvalue(x::ForwardDiff.Dual) = _rawvalue(ForwardDiff.value(x))

const _PS_MAXITER = 20
const _PS_TOL     = 1e-13

# Transverse shear correction κ_s = 5/6, matching LinearElastic.  Applied to the shear
# strain as γ → √κ_s·γ rather than to a stiffness: W is quadratic in γ about γ=0, so
# this is exactly the κ_s factor on the shear energy, and it enters through the single
# point every path funnels into — the explicit Cs (Hessian of W at γ=0) and the
# through-thickness energy quadrature therefore carry the same correction.
const _SQRT_KAPPA_S = sqrt(5/6)

# C₃₃ from the plane-stress condition S³³ = 2 ∂W/∂C₃₃ = 0, i.e. W stationary in C₃₃.
# Newton from the incompressible value; derivatives w.r.t. c/γ flow through the
# iteration, so the reduced energy stays exactly differentiable for the AD tangents.
# Convergence is quadratic (one step for energies quadratic in E), and the branch
# tests primal values only so nested duals are safe.  Non-convergence within
# _PS_MAXITER returns the last iterate rather than throwing, so a line search that
# probes a wild state still gets a (poor) value instead of an exception.
@inline function _C33_planestress(mat::Hyperelastic, c::SymmetricTensor{2,2}, γ₁, γ₂, det_A, Jinv)
    f(x)  = mat.W(_to_C_cart(build_C3D(c, γ₁, γ₂, x), Jinv))
    df(x) = ForwardDiff.derivative(f, x)
    C33 = get_C33(c, γ₁, γ₂, det_A)
    for _ in 1:_PS_MAXITER
        δ = df(C33) / ForwardDiff.derivative(df, C33)
        C33 -= δ
        abs(_rawvalue(δ)) ≤ _PS_TOL * (1 + abs(_rawvalue(C33))) && break
    end
    C33
end

# Through-thickness condensation: incompressibility (analytic) or plane stress (Newton).
@inline _C33(mat::Hyperelastic, c::SymmetricTensor{2,2}, γ₁, γ₂, det_A, Jinv) =
    mat.incompressible ? get_C33(c, γ₁, γ₂, det_A) : _C33_planestress(mat, c, γ₁, γ₂, det_A, Jinv)

# Evaluate W at the physical Cartesian C, no shear (γ=0).
@inline function _W_phys(mat::Hyperelastic, c::SymmetricTensor{2,2}, det_A, Jinv)
    z = zero(eltype(c))
    _W_phys(mat, c, z, z, det_A, Jinv)
end

# Evaluate W at the physical Cartesian C, with shear γ₁, γ₂ (scaled by √κ_s).
@inline function _W_phys(mat::Hyperelastic, c::SymmetricTensor{2,2}, γ₁, γ₂, det_A, Jinv)
    g₁ = _SQRT_KAPPA_S * γ₁; g₂ = _SQRT_KAPPA_S * γ₂
    C33 = _C33(mat, c, g₁, g₂, det_A, Jinv)
    mat.W(_to_C_cart(build_C3D(c, g₁, g₂, C33), Jinv))
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