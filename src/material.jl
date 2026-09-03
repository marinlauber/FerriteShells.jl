using Tensors
using ForwardDiff

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

# used to specialise the Hyperelastic formulation
abstract type Compressible end
abstract type Incompressible end
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
struct NeoHookean{T}
    μ::T
end
(w::NeoHookean)(C) = w.μ/2 * (tr(C) - 3)

μ = 80.0e3; t = 1.0e-3
mat = Hyperelastic(NeoHookean(μ), t)
```

Example — Mooney–Rivlin

```julia
struct MooneyRivlin{T}
    c₁::T
    c₂::T
end
(w::MooneyRivlin)(C) = w.c₁*(tr(C) - 3) + w.c₂*((tr(C)^2 - C ⊡ C)/2 - 3)

c₁ = 40.0e3; c₂ = 20.0e3; t = 1.0e-3
mat = Hyperelastic(MooneyRivlin(c₁, c₂), t)
```

Example — Saint-Venant–Kirchhoff (compressible, `ν = 0.3`), type-stable functor form

```julia
struct SaintVenantKirchhoff{T}
    λ::T
    μ::T
end
(w::SaintVenantKirchhoff)(C) = (Eg = (C - one(C))/2; w.λ/2 * tr(Eg)^2 + w.μ * (Eg ⊡ Eg))

E = 0.35e8; ν = 0.3; t = 0.2e-3
mat = Hyperelastic(SaintVenantKirchhoff(E*ν/((1+ν)*(1-2ν)), E/(2*(1+ν))), t; incompressible=false)
```
"""
struct Hyperelastic{I, F, T<:AbstractFloat} <: AbstractMaterial
    W              :: F
    thickness      :: T
    function Hyperelastic(W::F, thickness::T=one(Float64);
                          incompressible::Bool=true) where {F, T<:AbstractFloat}
        @assert thickness > 0 "Thickness must be positive"
        I = incompressible ? Incompressible : Compressible
        new{I, F, T}(W, thickness)
    end
end

# C₃₃ from det(C_nat) = det_A so that det(C_cart) = 1 (physical incompressibility).
# det_A = det(A_metric) = |A₁ × A₂|² (reference area element squared).
# Reduces to det_A/det₂(c) when γ=0 (KL / no-shear limit).
@inline get_C33(c::SymmetricTensor{2,2}, γ₁, γ₂, det_A) = (det_A + c[2,2]*γ₁^2 - 2*c[1,2]*γ₁*γ₂ + c[1,1]*γ₂^2) / det(c)
@inline function get_C33(mat::Hyperelastic{Incompressible}, c::SymmetricTensor{2,2}, γ₁, γ₂, det_A, Jinv)
    return (det_A + c[2,2]*γ₁^2 - 2*c[1,2]*γ₁*γ₂ + c[1,1]*γ₂^2) / det(c)
end

# Build the full 3×3 right Cauchy–Green tensor.
# SymmetricTensor{2,3} lower-triangle column-major storage: (C₁₁,C₁₂,C₁₃,C₂₂,C₂₃,C₃₃)
@inline function build_C3D(c::SymmetricTensor{2,2}, γ₁, γ₂, C33)
    TT = promote_type(eltype(c), typeof(C33))
    SymmetricTensor{2,3,TT}((TT(c[1,1]), TT(c[1,2]), TT(γ₁), TT(c[2,2]), TT(γ₂), TT(C33)))
end

# Reference Jacobian: columns = A₁, A₂, G₃ in Cartesian.  Stored column-major.
@inline J_ref(A₁, A₂, G₃) = Tensor{2,3}((A₁[1],A₁[2],A₁[3], A₂[1],A₂[2],A₂[3], G₃[1],G₃[2],G₃[3]))

# Transform C_nat (natural frame) → C_cart (Cartesian): C_cart = Jinv' C_nat Jinv.
@inline to_C_cart(C_nat::SymmetricTensor{2,3}, Jinv::Tensor{2,3}) = symmetric(Jinv' ⋅ Tensor{2,3}(C_nat) ⋅ Jinv)

# Strip every layer of ForwardDiff nesting — used to branch on primal values only.
@inline rawvalue(x::Real) = x
@inline rawvalue(x::ForwardDiff.Dual) = rawvalue(ForwardDiff.value(x))


# Transverse shear correction κ_s = 5/6
const κₛ = sqrt(5/6)

# C₃₃ from the plane-stress condition S³³ = 2 ∂W/∂C₃₃ = 0, i.e. W stationary in C₃₃.
@inline function get_C33(mat::Hyperelastic{Compressible}, c::SymmetricTensor{2,2}, γ₁, γ₂, det_A, Jinv)
    f(x)  = mat.W(to_C_cart(build_C3D(c, γ₁, γ₂, x), Jinv))
    df(x) = ForwardDiff.derivative(f, x)
    C33 = get_C33(c, γ₁, γ₂, det_A) # initial guess
    for _ in 1:20
        δ = df(C33) / ForwardDiff.derivative(df, C33)
        C33 -= δ
        abs(rawvalue(δ)) ≤ 1e-13 * (1 + abs(rawvalue(C33))) && break
    end
    C33
end

# Evaluate W at the physical Cartesian C, no shear (γ=0).
@inline function W_phys(mat::Hyperelastic, c::SymmetricTensor{2,2}, det_A, Jinv)
    z = zero(eltype(c))
    W_phys(mat, c, z, z, det_A, Jinv)
end

# Evaluate W at the physical Cartesian C, with shear γ₁, γ₂ (scaled by √κ_s).
@inline function W_phys(mat::Hyperelastic, c::SymmetricTensor{2,2}, γ₁, γ₂, det_A, Jinv)
    g₁ = κₛ * γ₁; g₂ = κₛ * γ₂
    C33 = get_C33(mat, c, g₁, g₂, det_A, Jinv)
    mat.W(to_C_cart(build_C3D(c, g₁, g₂, C33), Jinv))
end

# Membrane stress N and consistent tangent C via nested gradient of W_phys.
# N^{αβ} = 2t ∂W/∂C_{αβ}; factor 2 from Tensors.jl Mandel off-diagonal convention.
function membrane_stress_and_tangent(mat::Hyperelastic, c_ms::SymmetricTensor{2,2},
                                     A_metric, A₁, A₂, G₃)
    det_A = det(A_metric)
    Jinv  = inv(J_ref(A₁, A₂, G₃))
    ∇W(c) = gradient(x -> W_phys(mat, x, det_A, Jinv), c)
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
    Jinv  = inv(J_ref(A₁, A₂, G₃))
    W_sh(γ) = W_phys(mat, c_ms, γ[1], γ[2], det_A, Jinv)
    Cs_full = mat.thickness * hessian(W_sh, zero(Vec{2,T}))
    Cs = SymmetricTensor{2,2,T}((Cs_full[1,1], Cs_full[1,2], Cs_full[2,2]))
    return D, Cs
end