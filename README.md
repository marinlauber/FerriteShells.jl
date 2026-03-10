# FerriteShells.jl

Assemblers for shells in Ferrite.jl

This package provides helper functions to assemble the different terms in the weak form of most classical shell formulation — C⁰ Kirchhoff--Love linear, C⁰ Koiter (non-linear Kirchhoff-Love), Reissner-Mindlin and Naghi (non-linear Reissner-Mindlin) shells.
Specifically, this package provides helper function to assemble the classical membrane, bending and shear contribution to the residuals and the consistent tangent stiffness matrix.
Proper assembly of these different terms lead to the different formulation mentioned above.

We refer to the reader to the specific weak form for each of these shell and their numerical an implementation limitiations.

## 1. `ShellCellValues`

Since most wak forms use in shell analysis are specified by specializing the classical weak form to the curvilinear system of the mid-plane of the shell, classical continuum mechanics quantities, such as the deformation gradient tensor $\bf{F}$ change when expressed in curvilinear coordinates.
To help assemble these specific quantities, this package provides a new `ShellCellValues<:AbstractCellValues`, which behaves identically to Ferrite's `CellValues`, but hold covariant basis vector, metric tensors and surface Jacobian at the integration points.
```julia
structure ShellCellValues <: AbstractCellValues
    qr :: QR
    ...
    a_αβ ::
    A_αβ ::
    d_ ::
end
```

> [!WARNING]
> For now, the residual and consistent tangent construction use `ForwardDiff.gradient` and `ForwardDiff.hessian`, so simplify implementation. This will be slow on large meshes (>1000 elements), in the future explicit expressions should replace those. Medium term, we can start with the membrane term only for the Reissner-Mindlin shell.

### 1.1 Element choice

For Kirchhoff-Love shells, the Q9 element with `Lagrange{RefQuadrilateral, 2}` is required to capture the bending term. This is the $\mathcal{C}^0$ Kirchhoff-Love approach.
RM without MITC (mixed interpolation) or reduced integration suffers from shear locking. For Q4 (bilinear) with full 2×2 Gauss integration the locking parameter scales as $(h/t)²$. Even at $t/h = 1$ (which isn't a thin shell), Q4 full integration noticeably under-predicts deflection, making it hard to meet the 5% tolerance. Q8's quadratic shape functions introduce bubble-like modes that relieve the spurious shear constraint.
For Reissner-Mindlin you can use Q4 (`Lagrange{RefQuadrilateral, 1}()`) the Q8/Q9 choice only matters when you have thin-shell bending-dominated problems with small $t/h$.

For a flat plate, the RM shear constraint (zero shear = Kirchhoff limit) requires:
\[
γ₁ = φ₁ + ∂u₃/∂x = 0
γ₂ = φ₂ + ∂u₃/∂y = 0
\]
Q4 (bilinear):
- $u₃ ∈ {1, x, y, xy}$ → $∂u₃/∂x ∈ {1, y}$ (only constant + linear-in-y)
- $φ₁ ∈ {1, x, y, xy}$ — has 4 DOFs, but must match $∂u₃/∂x$ which has no x-dependence
- The xy term in $φ₁$ can't be independently absorbed — it couples to the bending modes and over-constrains them

Q8 (serendipity, quadratic): https://defelement.org/elements/examples/quadrilateral-lagrange-equispaced-2.html
- $u₃ ∈ {1, x, y, xy, x², y², x²y, xy²}$ → $∂u₃/∂x ∈ {1, y, x, y², xy, x²}$ — much richer
- $φ₁$ also quadratic, so both sides of $φ₁ + ∂u₃/∂x = 0$ live in the same polynomial space
- The constraint can be satisfied approximately without locking the bending modes

For a cantilever under tip load, the exact solution has $u₃ \sim x²(3L-x)$ (cubic) and $φ \sim x(L - x/2)$ (quadratic). Q4 can only represent linear variation within each element so the Kirchhoff constraint produces large spurious shear energy. Q8 can represent quadratic within each element, so it matches the constraint far more accurately.

The locking isn't eliminated in Q8 with full integration (it still locks for very small $t/h$), just pushed to much thinner regimes. For $t/h = 1$ as in our test, Q8 is essentially lock-free.

Why Q4 fails for KL:

The curvature tensor requires second derivatives of shape functions. For Q4 (bilinear $N \sim 1$, $ξ$, $η$, $ξη$):
- $∂²N/∂ξ² = 0$, $∂²N/∂η² = 0$ → $κ₁₁$ and $κ₂₂$ are always zero
- $∂²N/∂ξ∂η ≠ 0$ → only twist $κ₁₂$ is captured

So Q4 KL is missing two curvature components entirely — it's not a locking issue, it's a representability issue. You need at least Q8 or Q9 (both have $ξ²$, $η²$ terms with non-zero second derivatives).
Why Q9 is preferred over Q8 for KL:
Q9 adds the center node with mode $(1-ξ²)(1-η²)$, which is a richer bubble function. It gives better accuracy and a cleaner zero-mode spectrum (as the n_zero == 21 test verifies).

For RM:
Q9 is also fine — the center bubble mode doesn't cause any harm and actually gives slightly more kinematic flexibility. The reason the cantilever test uses
Q8 is purely practical: `generate_grid(QuadraticQuadrilateral, ...) gives Q8 directly, while Q9 requires manually constructing the mesh. No second derivatives of shape functions are needed in RM (curvature comes from the director field d), so the Q8 vs Q9 distinction for KL doesn't apply.

The real dividing line is Q4 vs quadratic (Q8/Q9):
- KL: Q4 is fundamentally broken (missing curvature components), Q8/Q9 both work
- RM: Q4 locks badly for bending, Q8/Q9 both work well

> [!WARNING]
> The RM formulation can still exhibit shear locking for very thin shells with Q4 elements. Use Q8 or Q9 for better performance.

> [!WARNING]
> The RM formulation assumes that the degrees of freedom are interleaved, which is different to Ferrite's ordering of DoFs when two different field are specified in the `DofHandler`
> ```julia
> dh = DofHandler(grid)
> add!(dh, :u, ip^3); add!(dh, :θ, ip^2)
> close!(dh)
> ```
> and use `shelldofs` to correctly index from the solution and in the assembler
>```julia
> for cell in CellIterator(dh)
>     ...
>     u_e = u[shelldofs(cell)]
>     ...
>     assemble!(assembler, shelldofs(cell), ke, re)
> end
> ```
> alternatively, you can bypass all this by specifying a a 5-dimensional unknown field to the `DofHandler` with `add!(dh, :u, ip^5)` which contains both the displacements and rotation at once, but this breaks some IO function from Ferrite.jl

## 2. General assemble template

This package provides helper functions to assemble the residual vector and consistent tangent stiffness matrix for linear and non-linear shell problems.
These helpers can use used with the rest of Ferrite.jl, and with correct setup of the interpolation space and element DOF and order.
An example of how these helpers are used in an assembly is shown below

```julia
function assemble_global_shell!(K, re, u, dh, scv::ShellCellValues, mat)
    ...
    asm = start_assemble(K, re)
    for cell in CellIterator(dh)
        # reset to  zero
        fill!(ke, 0.0); fill!(re, 0.0)
        # measure the cell, local covariant basis, metric tensor, etc.
        reinit!(scv, cell)
        # nonlinear contributions come from displacements
        x   = getcoordinates(cell)
        u_e = u[celldofs(cell)]
        # membrane contributions
        membrane_tangent!(ke, scv, x, u_e, mat)
        membrane_residuals!(re, scv, x, u_e, mat)
        # bending contribution
        bending_tangent!(ke, scv, x, u_e, mat)
        bending_residuals!(re, scv, x, u_e, mat)
        # shear contribution, with MITC9 here
        shear_tangent!(ke, scv, x, u_e, mat, :MITC9)
        shear_residuals!(re, scv, x, u_e, mat, :MITC9)
        # assemble in the global matrix
        assemble!(asm, celldofs(cell), ke, re)
    end
end
```

## 3. General virtual work for a solid in a curvilinear coordinate system

The key: virtual work is a scalar

The internal virtual work is a scalar contraction in the covariant frame:

\[
δW_\text{int} = ∫ S^{αβ} δE_{αβ} dA  +  ∫ M^{αβ} δκ_{αβ} dA
\]
where $S^{αβ}$ and $M^{αβ}$ are contravariant stress/moment resultants, and $E_{αβ}$, $κ_{αβ}$ are covariant strain/curvature measures. The double contraction is
coordinate-invariant — it gives the same scalar regardless of frame.

Where the "frame" lives

The DOFs $\bm{u}$ are Cartesian displacements. The covariant strains are functions of those Cartesian DOFs:
\[
E_{αβ} = E_{αβ}(\bm{u}),   κ_{αβ} = κ_{αβ}(\bm{u})
\]
The nodal residual force is the derivative of the scalar energy $W$ w.r.t. each Cartesian DOF:
\[
r_{I,i} = ∂W/∂u_{I,i} = ∫ S^{αβ}  (∂E_{αβ}/∂u_{I,i})  dA
\]
The chain rule is the back-transformation.

For the membrane, $\partial E_{αβ}/\partial u_{I,i}$ expands via the chain rule through the covariant base vectors:
\[
\partial E_{αβ}/\partial u_{I,i} = ½ (g_α · eᵢ · N_{I,β}  +  g_β · eᵢ · N_{I,α})
\]
where $g_α = ∂x/∂ξ^α$ are the Cartesian components of the surface tangents. Contracting with $S^{αβ}$ gives:
\[
r_{I,i} = ∫ (S^{αβ} g_α)ᵢ · N_{I,β} dA
\]
The factor $S^{αβ} g_α$ is the Cauchy stress vector expressed in Cartesian coordinates — the base vectors $g_α$ carry the covariant-to-Cartesian map. No explicit rotation matrix is ever applied; the geometry of the surface (encoded in $g_α$) provides the transformation automatically.
For the bending terms, ForwardDiff.gradient differentiates the scalar bending energy with respect to the flat Cartesian DOF vector, which gives the same result algorithmically.

Summary:

| Quantity    | Frame                                 |
| ----------- | ------------------------------------- |
| DOFs $\bm{u}$      | Cartesian (global)                    |
| $E_{αβ}$, $κ_{αβ}$  | Covariant (surface-local)             |
| $S^{αβ}$, $M^{αβ}$  | Contravariant (surface-local)         |
| Residuals $\bm{r}$ | Cartesian (via chain rule through $g_α$) |

The covariant quantities are intermediate values in a scalar energy computation. Differentiating that scalar with respect to Cartesian DOFs automatically produces Cartesian forces — the base vectors gα in the kinematics carry the implicit transformation.

### 3.1 Membrane residual and consistent tangent terms


#### 3.1.1 Membrane Residual Implementation

Residual for node I:

\[
R_I = \int \partial_\alpha N_I \, N_{\alpha\beta} \, a_\beta \, dA
\]


#### 3.1.2 Consistent Tangent Derivation

Residual linearization:

\[
\delta R_I = \int \partial_\alpha N_I ( \delta N_{\alpha\beta} a_\beta + N_{\alpha\beta} \delta a_\beta ) dA
\]

- **Material stiffness**: $δN_{αβ} a_β$ term
- **Geometric stiffness**: $N_{αβ} δa_β$ term
- **$δa_β = ∂_β N_J δu_J$**
- Tangent naturally splits into material + geometric contributions without B-matrix

> [!NOTE]
> Explicit vs B-Matrix
>- **B-matrix**: classical, hides geometry, geometric stiffness must be derived separately, large matrices
>- **Explicit variation**: continuum mechanics directly implemented, geometric stiffness emerges naturally, tensor contractions, better for geometrically exact shells

## 4. Contravariant Elasticity

As seen in ... the covariant surface measures are combined with the contravariant material tensors to form the weak form.
We must expresse the classical 4th order material tensors in this contravariant frame.

> [!NOTE]
> On a surface, we have: $a^{αβ}$ = inv($a_{αβ}$)
> To change from covariant to contravariant frame, Tensors.jl defines this inverse for any `inv(A::SymmetricTensor{2,2})`.

## 5. Examples

### 5.1 Patch test

Reference solution:

The standard Cook's membrane parameters are E = 1, ν = 1/3, t = 1, shear traction q = 1/16 (total shear force F = 1 on right edge of height 16). The reference quantity is the
vertical displacement at the top-right corner (48, 60):

| Source | v_tip |
| -- | -- |
| Converged (fine mesh)         | 23.9648 |
| Simo & Rifai 1990 (Q4, 32×32) | ≈ 23.96 |


Primary reference:
Simo, J.C. & Rifai, M.S., "A class of mixed assumed strain methods and the method of incompatible modes", IJNME 29, 1595–1638 (1990)

Original problem source:
Cook, R.D., "Improved two-dimensional finite element", Journal of the Structural Division, ASCE 100, 1851–1863 (1974)


### 5.2 Cook's membrane (KL + RM)

Linear elastic pure membrane problem.

### 5.3 Cantilever beam (RM)

1. Clamped BC cannot be fully imposed

    KL clamped BC requires both:
    - $u = 0$ (displacement) — imposable via Dirichlet on nodes ✓
    - $\partial u_3 / \partial n = 0$ (zero slope) — no DOF to constrain in $C^0$ KL ✗

    So the clamped end acts like a hinge (pinned) rather than a clamp — the structure is free to rotate about the clamped edge. This alone makes the cantilever
    problem mechanically wrong.

2. Spurious hinge modes at element interfaces

    The bending energy is computed from second derivatives within each element. A slope discontinuity (kink) at an element interface lies on a set of measure
    zero and contributes zero to the bending energy. Each interface therefore acts as a free hinge, adding spurious zero-energy modes — one per interface.

    For an N-element cantilever this means N−1 interior hinges plus the unconstrained rotation at the clamped end — effectively a mechanism.


### 5.3 Square Pillow inflation

The ODE formulation:
Differentiating the equilibrium constraint $R_\text{int}(u) = p \cdot F(u)$ w.r.t. the load parameter $p$:
\[
K_\text{eff} \cdot \frac{du}{dp} = F(u)
\]

where $K_\text{eff} = K_\text{membrane}(u) - K_\text{pressure\_tangent}(u, p)$. This is a first-order ODE in $u$ with $p$ as "time" — exactly what OrdinaryDiffEq solves. No Newton iterations needed; the ODE solver handles the path
following.

The singularity: At $u=0$ (flat), $z$-DOFs have zero stiffness so $K_\text{eff}$ is singular. A small initial $z$-perturbation satisfying the BCs is required.

## Contributing

