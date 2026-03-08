# FerriteShells.jl

Multiple shells in Ferrite.jl

```
FerriteShells.jl
│
├── src/
│   ├── FerriteShells.jl
│   │
│   ├── kinematics/
│   │   ├── midsurface.jl
│   │   ├── directors.jl
│   │   ├── strains.jl
│   │
│   ├── elements/
│   │   ├── abstract_shell.jl
│   │   ├── rm_linear_tri.jl
│   │   ├── rm_linear_quad.jl
│   │
│   ├── mitc/
│   │   ├── mitc_tri.jl
│   │   ├── mitc_quad.jl
│   │
│   ├── integration/
│   │   ├── shell_cellvalues.jl
│   │
│   ├── assembly/
│   │   ├── residual.jl
│   │   ├── tangent.jl
│   │
│   └── materials/
│       ├── linear_rm.jl
│       ├── hyperelastic_rm.jl
│
└── test/
    ├── patch_tests.jl
    ├── rigid_body_tests.jl
    ├── uniform_stretch_tests.jl
    └── ...
```

Claude code development
```
claude --resume 24b48640-280b-467c-853d-48a10578c8c3
```

## Membrane-Only Implementation Notes

A Ferrite.jl extension for Reissner-Mindlin shells, focusing initially on linear triangles/quadrilaterals with MITC shear integration. The emphasis is on a membrane-only, geometrically exact implementation, suitable for debugging and initial testing.

---

## 1. `ShellCellValues` Design

- **Purpose**: Store reference geometry, shape function data, and per-quadrature computation.
- **Philosophy**: Follow Ferrite.jl pattern; no current configuration storage per quadrature, only computed on demand.

### Structure
```julia
struct ShellCellValues{T,CV<:Ferrite.CellValues,M}
    cellvalues::CV
    thickness::T
    mitc::M
    Xnodal::Vector{Vec{3,T}}
end
```

- `reinit!(scv, cell, X)` stores reference nodal coordinates.
- `update!(scv, q, u_e, d_e)` computes current geometry and returns `ShellGeometry`.

### `ShellGeometry`
```julia
struct ShellGeometry{T}
    A1::Vec{3,T}
    A2::Vec{3,T}
    A_metric::SymmetricTensor{2,2,T}
    a1::Vec{3,T}
    a2::Vec{3,T}
    a_metric::SymmetricTensor{2,2,T}
    normal::Vec{3,T}
    detJ::T
    d::Vec{3,T}
    dd1::Vec{3,T}
    dd2::Vec{3,T}
end
```

---

## 2. Shell Kinematics Layer

- **Input**: `ShellGeometry`
- **Output**: `ShellStrain`
- **Membrane Strain**: Green–Lagrange: `E = 0.5 * (a_metric - A_metric)`
- **Bending Strain**: `K` computed from director gradients
- **Shear Strain**: `γ = d ⋅ a_α`

```julia
struct ShellStrain{T}
    E::SymmetricTensor{2,2,T}
    K::SymmetricTensor{2,2,T}
    γ::Vec{2,T}
end

function compute_shell_strain(geom)
    E = 0.5 * (geom.a_metric - geom.A_metric)
    K = compute_bending_strain(geom)
    γ = compute_raw_shear(geom)
    return ShellStrain(E, K, γ)
end
```

- Fully nonlinear
- Compatible with finite rotations and finite membrane strains
- Variations can be computed for Newton assembly without classical B-matrix

---

## 3. Explicit vs B-Matrix

- **B-matrix**: classical, hides geometry, geometric stiffness must be derived separately, large matrices
- **Explicit variation**: continuum mechanics directly implemented, geometric stiffness emerges naturally, tensor contractions, better for geometrically exact shells

**Conclusion**: Use explicit variation for nonlinear shells (especially director-based).

---

## 4. Membrane Residual Implementation

Residual for node I:

\[
R_I = \int \partial_\alpha N_I \, N_{\alpha\beta} \, a_\beta \, dA
\]

### Linear Isotropic Membrane Material
```julia
struct LinearMembraneMaterial{T}
    E::T
    ν::T
    thickness::T
end

function membrane_stress(material, E::SymmetricTensor{2,2,T}) where T
    E_mod = material.E
    ν = material.ν
    t = material.thickness
    factor = E_mod / (1 - ν^2)

    N11 = t*(factor*E[1,1] + factor*ν*E[2,2])
    N22 = t*(factor*ν*E[1,1] + factor*E[2,2])
    N12 = t*(factor*(1-ν)/2*2*E[1,2])

    return SymmetricTensor{2,2,T}(N11, N12, N22)
end
```

### Residual Assembly
```julia
function element_membrane_residual!(re, scv, u_e, material)
    fill!(re, 0.0)
    nqp = getnquadpoints(scv)
    for q in 1:nqp
        geom = update!(scv, q, u_e)
        a1, a2 = geom.a1, geom.a2
        E = 0.5*(geom.a_metric - geom.A_metric)
        N = membrane_stress(material, E)
        w = getweight(scv, q) * geom.detJ
        nn = getnnodes(scv)
        for I in 1:nn
            dN_dξ1 = shape_gradient(scv.cellvalues, I, q)[1]
            dN_dξ2 = shape_gradient(scv.cellvalues, I, q)[2]
            v = dN_dξ1*(N[1,1]*a1 + N[1,2]*a2) + dN_dξ2*(N[1,2]*a1 + N[2,2]*a2)
            re[3I-2:3I] .+= v * w
        end
    end
end
```

---

## 5. Consistent Tangent Derivation

Residual linearization:

\[
\delta R_I = \int \partial_\alpha N_I ( \delta N_{\alpha\beta} a_\beta + N_{\alpha\beta} \delta a_\beta ) dA
\]

- **Material stiffness**: δN_{αβ} a_β term  
- **Geometric stiffness**: N_{αβ} δa_β term  
- **δa_β = ∂_β N_J δu_J**  
- Tangent naturally splits into material + geometric contributions without B-matrix

### Tangent Pseudo-code
```julia
function element_membrane_tangent!(Ke, scv, u_e, material)
    fill!(Ke, 0.0)
    nqp = getnquadpoints(scv)
    nn = getnnodes(scv)
    for q in 1:nqp
        geom = update!(scv, q, u_e)
        a1, a2 = geom.a1, geom.a2
        E = 0.5*(geom.a_metric - geom.A_metric)
        N, C = membrane_stress_and_tangent(material, E)
        w = getweight(scv, q) * geom.detJ
        for I in 1:nn
            ∂NI1, ∂NI2 = shape_gradient(scv.cellvalues, I, q)...
            for J in 1:nn
                ∂NJ1, ∂NJ2 = shape_gradient(scv.cellvalues, J, q)...
                # Geometric stiffness
                Kgeo = (∂NI1*(N[1,1]*∂NJ1 + N[1,2]*∂NJ2) + ∂NI2*(N[1,2]*∂NJ1 + N[2,2]*∂NJ2)) * I₃
                # Material stiffness block (contract C with B-vectors)
                Kmat = compute_material_block(∂NI1, ∂NI2, ∂NJ1, ∂NJ2, a1, a2, C)
                Ke[3I-2:3I, 3J-2:3J] .+= (Kgeo + Kmat) * w
            end
        end
    end
end
```

---

## 6. Membrane Stress and Tangent Function

```julia
function membrane_stress_and_tangent(material::LinearMembraneMaterial{T},
                                     E::SymmetricTensor{2,2,T}) where T
    E_mod, ν, t = material.E, material.ν, material.thickness
    factor = E_mod / (1-ν^2)

    N11 = t*(factor*E[1,1] + factor*ν*E[2,2])
    N22 = t*(factor*ν*E[1,1] + factor*E[2,2])
    N12 = t*(factor*(1-ν)/2*2*E[1,2])

    N = SymmetricTensor{2,2,T}(N11, N12, N22)

    # 4th-order tangent (for explicit contraction)
    C = zeros(SymmetricTensor{2,2,SymmetricTensor{2,2,T}})
    C[1,1] = factor; C[1,2] = factor*ν
    C[2,1] = factor*ν; C[2,2] = factor
    C[3,3] = factor*(1-ν)/2

    return N, C
end
```


### Reference Solutions for Patch Tests

Reference solution:

The standard Cook's membrane parameters are E = 1, ν = 1/3, t = 1, shear traction q = 1/16 (total shear force F = 1 on right edge of height 16). The reference quantity is the
vertical displacement at the top-right corner (48, 60):

┌───────────────────────────────┬─────────┐
│            Source             │  v_tip  │
├───────────────────────────────┼─────────┤
│ Converged (fine mesh)         │ 23.9648 │
├───────────────────────────────┼─────────┤
│ Simo & Rifai 1990 (Q4, 32×32) │ ≈ 23.96 │
└───────────────────────────────┴─────────┘

Primary reference:
Simo, J.C. & Rifai, M.S., "A class of mixed assumed strain methods and the method of incompatible modes", IJNME 29, 1595–1638 (1990)

Original problem source:
Cook, R.D., "Improved two-dimensional finite element", Journal of the Structural Division, ASCE 100, 1851–1863 (1974)


### Square Pillow inflation
The ODE formulation:
Differentiating the equilibrium constraint R_int(u) = p · F(u) w.r.t. the load parameter p:

K_eff · du/dp = F(u)

where K_eff = K_membrane(u) − K_pressure_tangent(u, p). This is a first-order ODE in u with p as "time" — exactly what OrdinaryDiffEq solves. No Newton iterations needed; the ODE solver handles the path
following.

The singularity: At u=0 (flat), z-DOFs have zero stiffness so K_eff is singular. A small initial z-perturbation satisfying the BCs is required.