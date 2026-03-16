## Shell fomulations

## Shell theory comparison

| | Linear KL | Koiter | Reissner–Mindlin | Naghdi |
|---|---|---|---|---|
| **DOFs/node** | 3 (u₁,u₂,u₃) | 3 (u₁,u₂,u₃) | 5 (u₁,u₂,u₃,φ₁,φ₂) | 5 (u₁,u₂,u₃,φ₁,φ₂) |
| **Director** | implicit: **n** = **a**₁×**a**₂/‖·‖ | implicit: **n** = **a**₁×**a**₂/‖·‖ | additive: **d** = **G**₃+φ₁**T**₁+φ₂**T**₂, ‖**d**‖≠1 | Rodrigues: **d** = cos‖φ‖**G**₃+sinc‖φ‖(φ₁**T**₁+φ₂**T**₂), ‖**d**‖=1 |
| **Membrane strain** | linear: ½(**A**_α·**u**,β+**A**_β·**u**,α) | Green–Lagrange: ½(a_αβ−A_αβ) | linear: ½(**A**_α·**u**,β+**A**_β·**u**,α) | Green–Lagrange: ½(a_αβ−A_αβ) |
| **Bending strain** | κ_αβ = −u₃,αβ (flat ref.) | κ_αβ = b_αβ−B_αβ | ½(**A**_α·**d**,β+**A**_β·**d**,α)−B_αβ | ½(**a**_α·**d**,β+**a**_β·**d**,α)−B_αβ |
| **Transverse shear** | γ=0 (Kirchhoff) | γ=0 (Kirchhoff) | γ_α = **A**_α·**d** | γ_α = **a**_α·**d** |
| **Finite rotations** | no | yes | small only (‖φ‖≪1) | yes |
| **C¹ for bending** | yes | yes | no | no |
| **In FerriteShells** | — | ✓ `_KL` functions | — | ✓ `_RM` functions |

The key distinction between RM and Naghdi is which base vectors appear in the strain measures: RM uses the **reference** base vectors **A**_α (linearised around the reference configuration), while Naghdi replaces them with the **current** **a**_α everywhere, giving fully nonlinear strains. The director parametrisation (non-unit additive vs unit Rodrigues) is a separate but related choice — in practice the two always appear together.

Koiter has no director DOFs; the normal is always implicit from the surface geometry, so the Kirchhoff constraint (zero shear) is built in and C¹ continuity is required for bending.

Classical RM (additive director, ‖**d**‖≠1) is not implemented; the `_RM` functions go directly to the geometrically exact Naghdi form via Rodrigues parametrisation.

### Kirchhoff-Love / Koiter shell

The Kirchhoff-Love kinematic assumption
$$
\Phi(\xi^1,\xi^2,\xi^3) = \phi(\xi^1,\xi^2) + \xi^3\vec{a}_3(\xi^1,\xi^2)
$$

### Reissner-Mindlin / Naghdi shell

The Reissner-Mindlin kinematic assumption
$$
\Phi(\xi^1,\xi^2,\xi^3) = \phi(\xi^1,\xi^2) + \xi^3\theta^\lambda(\xi^1,\xi^2) \vec{a}_\lambda(\xi^1,\xi^2)
$$

### 1.1 Element choice

For Kirchhoff-Love shells, the Q9 element with `Lagrange{RefQuadrilateral, 2}` is required to capture the bending term. This is the $\mathcal{C}^0$ Kirchhoff-Love approach.
RM without MITC (mixed interpolation) or reduced integration suffers from shear locking. For Q4 (bilinear) with full 2×2 Gauss integration the locking parameter scales as $(h/t)²$. Even at $t/h = 1$ (which isn't a thin shell), Q4 full integration noticeably under-predicts deflection, making it hard to meet the 5% tolerance. Q8's quadratic shape functions introduce bubble-like modes that relieve the spurious shear constraint.
For Reissner-Mindlin you can use Q4 (`Lagrange{RefQuadrilateral, 1}()`) the Q8/Q9 choice only matters when you have thin-shell bending-dominated problems with small $t/h$.

For a flat plate, the RM shear constraint (zero shear = Kirchhoff limit) requires:
$$
\gamma_1 = \psi_1 + \frac{∂ u_3}{∂ x} = 0
\gamma_2 = \psi_2 + \frac{∂ u_3}{∂ y} = 0
$$
Q4 (bilinear):
- $u_3 ∈ {1, x, y, xy}$ → $∂u_3/∂x ∈ {1, y}$ (only constant + linear-in-y)
- $\psi_1 ∈ {1, x, y, xy}$ — has 4 DOFs, but must match $∂u_3/∂x$ which has no x-dependence
- The xy term in $\psi_1$ can't be independently absorbed — it couples to the bending modes and over-constrains them

Q8 (serendipity, quadratic): https://defelement.org/elements/examples/quadrilateral-lagrange-equispaced-2.html
- $u_3 ∈ {1, x, y, xy, x², y², x²y, xy²}$ → $∂u₃/∂x ∈ {1, y, x, y², xy, x²}$ — much richer
- $\psi_1$ also quadratic, so both sides of $\psi_1 + ∂u₃/∂x = 0$ live in the same polynomial space
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
Q9 is also fine — the center bubble mode doesn't cause any harm and actually gives slightly more kinematic flexibility. The reason the cantilever test uses Q9 is purely practical: `generate_grid(QuadraticQuadrilateral, ...) gives Q9 directly, while Q8 requires manually constructing the mesh. No second derivatives of shape functions are needed in RM (curvature comes from the director field d), so the Q8 vs Q9 distinction for KL doesn't apply.

In practice Q9 (what the code already uses) is generally preferred over Q8 — it has the center node which gives better accuracy for bending and avoids the parasitic zero-energy modes that Q8 can exhibit with reduced integration.

The real dividing line is Q4 vs quadratic (Q8/Q9):
- KL: Q4 is fundamentally broken (missing curvature components), Q8/Q9 both work
- RM: Q4 locks badly for bending, Q8/Q9 both work well

To improve the KL formulation:
 - [ ] Discrete Kirchhoff elements (DKQ) which enforce the KL constraint only at discrete boundary points
 - [ ] C1 elements (Hermite, NURBS/IGA)
 - [ ] MITC/mixed formulations

The core problem: KL bending energy W = ½∫ κ:D:κ dA requires integrating κ_αβ = -u₃,αβ — second derivatives — which are Dirac deltas at element boundaries with C0 elements.
Mixed formulation (Hellinger-Reissner) introduce the moment tensor M_αβ as an independent field and rewrite the energy:

W = ∫ (M_αβ κ_αβ - ½ M_αβ D⁻¹_αβγδ M_γδ) dA

Then integrate the M:κ term by parts:

∫ M_αβ κ_αβ dA = ∫ M_αβ (-u₃,αβ) dA = ∫ M_αβ,β u₃,α dA - [boundary terms]

The second derivative of u₃ has been shifted onto a first derivative of M. Now:
- u₃ only needs C0 (first derivatives square-integrable)
- M_αβ needs to be C0 (first derivatives square-integrable)
- No second derivatives of u₃ appear in the weak form at all

The cost is extra DOFs for M_αβ and a saddle-point system (inf-sup condition must be satisfied for stability).

MITC approach

MITC avoids the extra M DOFs by doing the decoupling differently: instead of computing κ_αβ pointwise from u₃,αβ, it samples κ at specific tying points along element edges and then interpolates those sampled values over the element using a reduced strain basis. The crucial effect is that the assumed strain field is no longer derived directly from the second derivatives of u₃ — it's interpolated independently, so the inter-element normal discontinuity doesn't pollute the integral. You get approximate curvature fields that are consistent with the displacement DOFs at the tying points only (weak consistency), not everywhere.

The conceptual link

Both approaches break the same chain: standard KL requires κ → u₃,αβ → C1. Mixed formulations break it by shifting derivatives from u₃ to M via integration by parts. MITC breaks it by decoupling the strain field from displacement derivatives entirely, sampling only at chosen points. Neither requires global C1 — they just relocate where smoothness is required.

The discrete Kirchhoff (DKQ) approach is related but different: it enforces γ = 0 (zero shear = Kirchhoff constraint) only at a finite set of points on element edges, which implicitly constructs a compatible normal rotation field without C1.


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

## 3. Linear solver strategies

Each Newton iteration requires solving one or two linear systems with the same stiffness matrix K_eff (two for the bordering method used in displacement-controlled path following). The cost breakdown matters because for large shell problems the linear solve — not the assembly — becomes the bottleneck.

### 3.1 Direct solvers (UMFPACK)

Julia's default sparse solver is UMFPACK (via SuiteSparse). A factorisation has two phases:

- **Symbolic**: analyses the sparsity pattern, computes a fill-reducing column permutation (e.g. AMD/COLAMD). Cost: 10–30% of total, depends only on the pattern.
- **Numeric**: factorises the actual values. Cost: 70–90% of total, must be repeated when values change.

In a Newton loop on a fixed mesh the sparsity pattern never changes. The three-argument form `lu!(F::UmfpackLU, K)` refactorises numerically while reusing the symbolic analysis from a previous `lu(K)` call. Combined with updating K_eff values in-place (exploiting the fact that K_int and K_pres share the same pattern), this avoids all redundant work:

```julia
# Outside the Newton loop — one-time setup:
K_eff = allocate_matrix(dh)           # same sparsity pattern as K_int, K_pres
assemble_all!(K_int, ..., u0, mat)
K_eff.nzval .= K_int.nzval            # initial values (p=0)
apply_zero!(K_eff, rhs_dummy, ch)
F_lu = lu(K_eff)                      # symbolic + numeric factorisation

# Inside the Newton loop:
K_eff.nzval .= K_int.nzval .- p .* K_pres.nzval   # update values in-place
apply_zero!(K_eff, rhs, ch)
lu!(F_lu, K_eff)                      # numeric only — reuses symbolic analysis
ldiv!(v1, F_lu, rhs1)                 # two back-substitutions, one factorisation
ldiv!(v2, F_lu, F_p)
```

The `.nzval .= ...` trick is valid because K_int, K_pres and K_eff all come from `allocate_matrix(dh)` and therefore have identical sparsity structure (same column pointers and row indices).

Cost comparison per Newton step:

| Pattern | Symbolic factorisations | Numeric factorisations | Back-substitutions |
|---|---|---|---|
| `K \ rhs` twice | 2 | 2 | 2 |
| `lu(K)` then `ldiv!` twice | 1 | 1 | 2 |
| `lu!(F, K)` then `ldiv!` twice | **0** | 1 | 2 |

### 3.2 Iterative solvers

For meshes beyond ~10⁵ DOFs, direct solvers become memory-limited (LU fill-in scales as O(N^1.5) for 2D problems). Iterative solvers avoid this but require a good preconditioner — shell problems are notoriously ill-conditioned.

**Condition number**: K_eff mixes membrane stiffness (~Et/L²) and bending stiffness (~Et³/L⁴), giving

$$\kappa(K_\text{eff}) \sim \frac{Et/L^2}{Et^3/L^4} = \left(\frac{L}{t}\right)^2$$

For t/L = 10⁻³ this is κ ~ 10⁶; unpreconditioned GMRES or CG would need O(κ) iterations.

**Practical options** (Julia packages):

| Package | Method | When suitable |
|---|---|---|
| [`IterativeSolvers.jl`](https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl) | GMRES, BiCGSTAB | Needs a preconditioner; entry point for experimentation |
| [`IncompleteLU.jl`](https://github.com/haampie/IncompleteLU.jl) | ILU(k) preconditioner | Moderate t/L (≥ 0.01); fill level k=1–2 usually sufficient |
| [`AlgebraicMultigrid.jl`](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl) | Ruge–Stüben AMG | Membrane-dominated problems; needs near-null-space vectors (6 rigid-body modes) |
| [`Pardiso.jl`](https://github.com/JuliaSparse/Pardiso.jl) | MKL Pardiso (direct) | Drop-in replacement for UMFPACK; faster for large problems, supports multithreading |
| [`MUMPS.jl`](https://github.com/JuliaSparse/MUMPS.jl) | MUMPS (direct, MPI) | Distributed memory; relevant for > 10⁶ DOFs |

For the thin-shell regime (t/L ≤ 10⁻²) standard AMG struggles because the near-null space contains both translational and rotational rigid-body modes, and the bending/membrane stiffness ratio spans many orders of magnitude. Purpose-built shell AMG (e.g. BoomerAMG in Hypre with proper near-null-space injection via [`Hypre.jl`](https://github.com/JuliaParallel/Hypre.jl)) is required to achieve mesh-independent convergence.

For the problem sizes typical in this package (n ≤ 32, DOFs ≤ 50k) UMFPACK with symbolic reuse is faster than any iterative alternative.

## 4. General virtual work for a solid in a curvilinear coordinate system

The key: virtual work is a scalar

The internal virtual work is a scalar contraction in the covariant frame:

$$
δW_\text{int} = ∫ S^{αβ} δE_{αβ} dA  +  ∫ M^{αβ} δκ_{αβ} dA
$$
where $S^{αβ}$ and $M^{αβ}$ are contravariant stress/moment resultants, and $E_{αβ}$, $κ_{αβ}$ are covariant strain/curvature measures. The double contraction is
coordinate-invariant — it gives the same scalar regardless of frame.

Where the "frame" lives

The DOFs $\bm{u}$ are Cartesian displacements. The covariant strains are functions of those Cartesian DOFs:
$$
E_{αβ} = E_{αβ}(\bm{u}),   κ_{αβ} = κ_{αβ}(\bm{u})
$$
The nodal residual force is the derivative of the scalar energy $W$ w.r.t. each Cartesian DOF:
$$
r_{I,i} = ∂W/∂u_{I,i} = ∫ S^{αβ}  (∂E_{αβ}/∂u_{I,i})  dA
$$
The chain rule is the back-transformation.

For the membrane, $\partial E_{αβ}/\partial u_{I,i}$ expands via the chain rule through the covariant base vectors:
$$
\partial E_{αβ}/\partial u_{I,i} = ½ (g_α · eᵢ · N_{I,β}  +  g_β · eᵢ · N_{I,α})
$$
where $g_α = ∂x/∂ξ^α$ are the Cartesian components of the surface tangents. Contracting with $S^{αβ}$ gives:
$$
r_{I,i} = ∫ (S^{αβ} g_α)ᵢ · N_{I,β} dA
$$
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

$$
R_I = \int \partial_\alpha N_I \, N_{\alpha\beta} \, a_\beta \, dA
$$


#### 3.1.2 Consistent Tangent Derivation

Residual linearization:

$$
\delta R_I = \int \partial_\alpha N_I ( \delta N_{\alpha\beta} a_\beta + N_{\alpha\beta} \delta a_\beta ) dA
$$

- **Material stiffness**: $δN_{αβ} a_β$ term
- **Geometric stiffness**: $N_{αβ} δa_β$ term
- **$δa_β = ∂_β N_J δu_J$**
- Tangent naturally splits into material + geometric contributions without B-matrix

> [!NOTE]
> Explicit vs B-Matrix
>- **B-matrix**: classical, hides geometry, geometric stiffness must be derived separately, large matrices
>- **Explicit variation**: continuum mechanics directly implemented, geometric stiffness emerges naturally, tensor contractions, better for geometrically exact shells

## 5. Contravariant Elasticity

As seen in ... the covariant surface measures are combined with the contravariant material tensors to form the weak form.
We must expresse the classical 4th order material tensors in this contravariant frame.

> [!NOTE]
> On a surface, we have: $a^{αβ}$ = inv($a_{αβ}$)
> To change from covariant to contravariant frame, Tensors.jl defines this inverse for any `inv(A::SymmetricTensor{2,2})`.

## 6. Examples

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

$$
K_\text{eff} \cdot \frac{du}{dp} = F(u)
$$

where $K_\text{eff} = K_\text{membrane}(u) - K_\text{pressure\_tangent}(u, p)$. This is a first-order ODE in $u$ with $p$ as "time" — exactly what OrdinaryDiffEq solves. No Newton iterations needed; the ODE solver handles the path
following.

The singularity: At $u=0$ (flat), $z$-DOFs have zero stiffness so $K_\text{eff}$ is singular. A small initial $z$-perturbation satisfying the BCs is required.

## 7. Kirchhoff–Love Limitations on Curved Shells

### 6.1 The C⁰ continuity problem

The current KL bending implementation computes the curvature change as

$$
\kappa_{\alpha\beta} = b_{\alpha\beta} - B_{\alpha\beta}
$$

where $b_{\alpha\beta} = \mathbf{x}_{,\alpha\beta} \cdot \mathbf{n}$ and $B_{\alpha\beta} = \mathbf{X}_{,\alpha\beta} \cdot \mathbf{N}$ are the current and reference second fundamental forms, computed from the within-element second derivatives of position.
This is mathematically correct inside each element. The problem is inter-element continuity.

Q8/Q9 elements (and any $C^0$ element) enforce continuity of position but not of the tangent vectors across element boundaries. The surface normal $\mathbf{n}$ is therefore **discontinuous at element edges**. For a curved shell, this means:

- The curvature $\kappa_{\alpha\beta}$ is computed independently inside each element with no coupling to neighboring normals.
- Bending moments have no mechanism to transfer across element boundaries — each interface effectively acts as a free hinge with zero bending energy.
- The global bending stiffness is systematically under-integrated, producing a structure that is too flexible.

For **flat shells** the normal is $(0,0,1)$ everywhere and discontinuities do not arise, so the formulation gives correct results. For **curved shells** the issue is fundamental.

### 6.2 Benchmark evidence

The Scordelis-Lo roof and pinched cylinder are standard curved-shell benchmarks. The RM formulation (which avoids the problem via explicit rotation DOFs) converges correctly; KL diverges:

| Benchmark | Reference | KL 4×4 | KL 8×8 | KL 16×16 | RM 16×16 |
|---|---|---|---|---|---|
| Scordelis-Lo $u_y$ | −0.3024 | −0.324 | −0.449 | −0.468 | −0.297 |
| Pinched cylinder $u_z$ | −1.825×10⁻⁵ | −6.1×10⁻⁵ | −1.5×10⁻⁴ | −1.8×10⁻⁴ | −1.7×10⁻⁵ |

The KL results diverge away from the reference as the mesh refines (the Scordelis-Lo coarse mesh accidentally undershoots near the reference, then overshoots as $h \to 0$). This is not a convergence rate issue — the method is converging to the wrong answer.

### 6.3 Planned improvements

Three approaches can fix this, in order of increasing complexity:

**Discrete Kirchhoff elements (DKQ/DKT)**

Instead of enforcing the KL constraint (zero transverse shear $\gamma_\alpha = 0$) everywhere, it is enforced only at a finite number of points along element edges. The rotation DOFs $\phi$ are introduced and then eliminated via the Kirchhoff constraint at the tying points, leaving only displacement DOFs. This gives:
- $C^0$ elements with correct inter-element bending continuity
- No additional global DOFs compared to standard KL
- Well-understood and widely used in engineering codes

This is the most practical near-term fix.

**MITC / assumed strain**

Instead of computing $\kappa_{\alpha\beta}$ pointwise from $u_{3,\alpha\beta}$, sample $\kappa$ at specific tying points along element edges and interpolate using a reduced strain basis. The assumed strain field is decoupled from the displacement second derivatives — the inter-element normal discontinuity no longer pollutes the integral.

Alternatively, the mixed (Hellinger–Reissner) formulation introduces the moment tensor $M_{\alpha\beta}$ as an independent field and rewrites the energy:

$$
W = \int \left( M_{\alpha\beta} \kappa_{\alpha\beta} - \tfrac{1}{2} M_{\alpha\beta} D^{-1}_{\alpha\beta\gamma\delta} M_{\gamma\delta} \right) dA
$$

Integration by parts shifts the second derivative from $u_3$ onto $M$:

$$
\int M_{\alpha\beta} \kappa_{\alpha\beta} \, dA = \int M_{\alpha\beta,\beta} \, u_{3,\alpha} \, dA - [\text{boundary terms}]
$$

Now only first derivatives of $u_3$ appear — $C^0$ is sufficient. The cost is extra DOFs for $M_{\alpha\beta}$ and a saddle-point system that must satisfy an inf-sup condition.

**Isogeometric analysis (IGA / NURBS)**

NURBS basis functions are globally $C^{p-1}$ continuous for degree $p$. Quadratic NURBS ($p=2$) are $C^1$, which is exactly what KL requires. IGA gives:
- Exact geometry representation for common curved shells (cylinders, spheres)
- Full KL bending continuity across patch boundaries
- No special element formulation needed

The main challenge is multi-patch coupling at $C^0$ or $C^1$ junctions, which requires additional constraint enforcement.


### Reissner–Mindlin / Naghdi Shell: Explicit Residual Derivation
---
1. Reference geometry
The shell midsurface is parameterized by convective coordinates (ξ¹, ξ²). The reference configuration is:
Covariant base vectors:
$$\mathbf{A}_\alpha = \frac{\partial \mathbf{X}}{\partial \xi^\alpha}, \quad \alpha = 1,2$$

Reference unit normal (director):
$$\mathbf{G}_3 = \frac{\mathbf{A}_1 \times \mathbf{A}_2}{|\mathbf{A}_1 \times \mathbf{A}_2|}$$

Reference metric:
$$A_{\alpha\beta} = \mathbf{A}_\alpha \cdot \mathbf{A}_\beta, \quad A = \det(A_{\alpha\beta}), \quad \sqrt{A} = |\mathbf{A}_1
\times \mathbf{A}_2|$$

Contravariant base vectors $\mathbf{A}^\alpha$ defined by $\mathbf{A}^\alpha \cdot \mathbf{A}_\beta = \delta^\alpha_\beta$, with
contravariant metric $A^{\alpha\beta} = \mathbf{A}^\alpha \cdot \mathbf{A}^\beta = (A_{\alpha\beta})^{-1}$.

Reference second fundamental form:
$$B_{\alpha\beta} = \mathbf{G}_3 \cdot \mathbf{A}_{\alpha,\beta}$$

Note: $\mathbf{A}_\alpha \cdot \mathbf{G}_{3,\beta} = -B_{\alpha\beta}$ (follows from differentiating $\mathbf{A}_\alpha \cdot
\mathbf{G}_3 = 0$).

---
2. Kinematic assumption

The 3D position of a material point in the deformed body:
$$\mathbf{x}(\xi^1, \xi^2, \xi^3) = \underbrace{\mathbf{X}(\xi^1,\xi^2) + \mathbf{u}(\xi^1,\xi^2)}_{\boldsymbol{\varphi}:
\text{ deformed midsurface}} + \xi^3 \mathbf{d}(\xi^1,\xi^2)$$

with $\xi^3 \in [-h/2, h/2]$ the through-thickness coordinate.

Assumptions:
- Straight fiber: The fiber through each midsurface point remains straight after deformation (no warping).
- Inextensible director: $|\mathbf{d}| = 1$ — the fiber does not stretch through the thickness ($E_{33} = 0$).
- Independent director: $\mathbf{d}$ is independent of $\mathbf{u}$ — it has its own DOFs (the rotations $\phi_1, \phi_2$).
This is what distinguishes RM/Naghdi from Kirchhoff–Love.

The reference body is parameterized by:
$$\mathbf{X}(\xi^1,\xi^2,\xi^3) = \mathbf{X}(\xi^1,\xi^2) + \xi^3 \mathbf{G}_3(\xi^1,\xi^2)$$

---
3. Deformed geometry

Deformed midsurface base vectors:
$$\mathbf{a}_\alpha = \frac{\partial \boldsymbol{\varphi}}{\partial \xi^\alpha} = \mathbf{A}_\alpha + \mathbf{u}_{,\alpha}$$

3D deformed covariant base vectors:
$$\mathbf{g}_\alpha = \frac{\partial \mathbf{x}}{\partial \xi^\alpha} = \mathbf{a}_\alpha + \xi^3 \mathbf{d}_{,\alpha}, \qquad
\mathbf{g}_3 = \frac{\partial \mathbf{x}}{\partial \xi^3} = \mathbf{d}$$

Reference 3D base vectors:
$$\mathbf{G}_\alpha = \mathbf{A}_\alpha + \xi^3 \mathbf{G}_{3,\alpha} \qquad \mathbf{G}_3$$

---
4. Green–Lagrange strains and shell truncation

The 3D Green–Lagrange strain components in the convected frame:
$$2E_{ij} = \mathbf{g}_i \cdot \mathbf{g}_j - \mathbf{G}_i \cdot \mathbf{G}_j$$

In-plane components ($\alpha,\beta = 1,2$):

$$2E_{\alpha\beta} = \mathbf{g}_\alpha \cdot \mathbf{g}_\beta - \mathbf{G}_\alpha \cdot \mathbf{G}_\beta$$
$$= (\mathbf{a}_\alpha + \xi^3 \mathbf{d}_{,\alpha}) \cdot (\mathbf{a}_\beta + \xi^3 \mathbf{d}_{,\beta}) - (\mathbf{A}_\alpha +
\xi^3 \mathbf{G}_{3,\alpha}) \cdot (\mathbf{A}_\beta + \xi^3 \mathbf{G}_{3,\beta})$$
$$= \underbrace{(a_{\alpha\beta} - A_{\alpha\beta})}{\text{order 1}} + \xi^3 \underbrace{(\mathbf{a}_\alpha \cdot
\mathbf{d}_{,\beta} + \mathbf{a}_\beta \cdot \mathbf{d}_{,\alpha} - \mathbf{A}_\alpha \cdot \mathbf{G}_{3,\beta} - \mathbf{A}_\beta
\cdot \mathbf{G}_{3,\alpha})}{\text{order } \xi^3} + \underbrace{(\xi^3)^2(\mathbf{d}_{,\alpha} \cdot \mathbf{d}_{,\beta} -
\mathbf{G}_{3,\alpha} \cdot \mathbf{G}_{3,\beta})}_{\text{neglected}}$$

Assumption: Truncate at $(\xi^3)^1$ — the Love–Kirchhoff hypothesis on the strain field (not on kinematics). Valid for $h \ll
R$ (radius of curvature) and $h \ll L$ (characteristic length).

This gives the shell strain decomposition:

$$2E_{\alpha\beta} \approx 2\varepsilon_{\alpha\beta} + 2\xi^3 \kappa_{\alpha\beta}$$

with:

$$\boxed{\varepsilon_{\alpha\beta} = \tfrac{1}{2}(a_{\alpha\beta} - A_{\alpha\beta})}$$

$$\boxed{\kappa_{\alpha\beta} = \tfrac{1}{2}(\mathbf{a}_\alpha \cdot \mathbf{d}_{,\beta} + \mathbf{a}_\beta \cdot
\mathbf{d}_{,\alpha}) - \tfrac{1}{2}(\mathbf{A}_\alpha \cdot \mathbf{G}_{3,\beta} + \mathbf{A}_\beta \cdot
\mathbf{G}_{3,\alpha})}$$

Using $\mathbf{A}_\alpha \cdot \mathbf{G}_{3,\beta} = -B_{\alpha\beta}$:
$$\kappa_{\alpha\beta} = \tfrac{1}{2}(\mathbf{a}\alpha \cdot \mathbf{d}{,\beta} + \mathbf{a}\beta \cdot \mathbf{d}{,\alpha}) -
B_{\alpha\beta}$$

Verification: at the reference state $\mathbf{d} = \mathbf{G}_3$, $\mathbf{a}_\alpha = \mathbf{A}_\alpha$: $\kappa_{\alpha\beta} =
\frac{1}{2}(\mathbf{A}_\alpha \cdot \mathbf{G}_{3,\beta} + \mathbf{A}_\beta \cdot \mathbf{G}_{3,\alpha}) - B_{\alpha\beta} =
-B_{\alpha\beta} - B_{\alpha\beta}/... $

Wait — more carefully: $\frac{1}{2}(\mathbf{A}_\alpha \cdot \mathbf{G}_{3,\beta} + \mathbf{A}_\beta \cdot \mathbf{G}_{3,\alpha}) =
\frac{1}{2}(-B_{\alpha\beta} - B_{\beta\alpha}) = -B_{\alpha\beta}$. So $\kappa_{\alpha\beta}|\text{ref} = -B_{\alpha\beta} -
B_{\alpha\beta}$... this is not zero?

This is the sign convention issue. The formula $-B_{\alpha\beta}$ in scv.B corresponds to $B_{\alpha\beta}^\text{code} =
-B_{\alpha\beta}^\text{standard}$, so with the code convention $\kappa = 0$ at reference. For this derivation, take
$B_{\alpha\beta}$ as defined by $\kappa_{\alpha\beta}|_\text{ref} = 0$, i.e.:

$$\kappa_{\alpha\beta} = \tfrac{1}{2}(\mathbf{a}_\alpha \cdot \mathbf{d}_{,\beta} + \mathbf{a}_\beta \cdot \mathbf{d}_{,\alpha}) -
B_{\alpha\beta}, \qquad B_{\alpha\beta} \equiv \tfrac{1}{2}(\mathbf{A}_\alpha \cdot \mathbf{G}_{3,\beta} + \mathbf{A}_\beta
\cdot \mathbf{G}_{3,\alpha}) = \mathbf{A}_\alpha \cdot \mathbf{G}_{3,\beta}$$
Transverse shear components ($E_{\alpha 3}$):

$$2E_{\alpha 3} = \mathbf{g}_\alpha \cdot \mathbf{g}_3 - \mathbf{G}_\alpha \cdot \mathbf{G}_3$$
$$= (\mathbf{a}_\alpha + \xi^3 \mathbf{d}_{,\alpha}) \cdot \mathbf{d} - (\mathbf{A}_\alpha + \xi^3 \mathbf{G}_{3,\alpha}) \cdot
\mathbf{G}_3$$

Using $|\mathbf{d}|=1 \Rightarrow \mathbf{d} \cdot \mathbf{d}_{,\alpha} = 0$, and $|\mathbf{G}_3|=1 \Rightarrow \mathbf{G}_3
\cdot \mathbf{G}_{3,\alpha}=0$, and $\mathbf{A}_\alpha \cdot \mathbf{G}_3 = 0$:

$$\boxed{\gamma_\alpha \equiv 2E_{\alpha 3} = \mathbf{a}_\alpha \cdot \mathbf{d}}$$

Verification: at reference: $\gamma_\alpha|\text{ref} = \mathbf{A}_\alpha \cdot \mathbf{G}_3 = 0$ ✓

Thickness strain $E_{33} = \frac{1}{2}(\mathbf{d}\cdot\mathbf{d} - \mathbf{G}_3\cdot\mathbf{G}_3) = \frac{1}{2}(1-1) = 0$ by
director inextensibility.

---
5. Constitutive law and stress resultants

Plane stress ($S^{33} = 0$), linear elastic. The second Piola–Kirchhoff stress resultants are obtained by integrating through
the thickness.

Contravariant elasticity tensor (plane stress, in the covariant basis):

$$\mathbb{C}^{\alpha\beta\gamma\delta} = \frac{E\nu}{1-\nu^2} A^{\alpha\beta}A^{\gamma\delta} +
\frac{E}{2(1+\nu)}\left(A^{\alpha\gamma}A^{\beta\delta} + A^{\alpha\delta}A^{\beta\gamma}\right)$$

Membrane force resultant ($t$-integrated):
$$N^{\alpha\beta} = t \mathbb{C}^{\alpha\beta\gamma\delta} \varepsilon_{\gamma\delta}$$

Bending moment resultant ($t^3/12$-integrated):
$$M^{\alpha\beta} = \frac{t^3}{12} \mathbb{C}^{\alpha\beta\gamma\delta} \kappa_{\gamma\delta}$$

Transverse shear resultant (with shear correction $\kappa_s = 5/6$):
$$Q^\alpha = \kappa_s \frac{Et}{2(1+\nu)} A^{\alpha\beta} \gamma_\beta$$

Note that all resultants are contravariant — they pair naturally with the covariant strain variations
$\delta\varepsilon_{\alpha\beta}$, $\delta\kappa_{\alpha\beta}$, $\delta\gamma_\alpha$.

---
6. Virtual work and residual

The internal virtual work for a displacement variation $\delta\mathbf{u}$ and director variation $\delta\mathbf{d}$:

$$\delta W_\text{int} = \int_\Omega \left[ N^{\alpha\beta} \delta\varepsilon_{\alpha\beta} + M^{\alpha\beta}
\delta\kappa_{\alpha\beta} + Q^\alpha \delta\gamma_\alpha \right] \sqrt{A} d\Omega$$

The residual is $\delta W_\text{int} - \delta W_\text{ext} = 0$ for all admissible variations.

Discretize with shape functions $N_I(\xi^1,\xi^2)$:
$$\mathbf{u} = \sum_I N_I \mathbf{u}_I, \qquad \mathbf{d} = \sum_I N_I \mathbf{d}_I(\boldsymbol{\phi}_I)$$

where $\mathbf{d}_I$ is the director at node $I$ parametrized by rotations $\boldsymbol{\phi}_I = (\phi_1^I, \phi_2^I)$ via
Rodrigues (Section 8). The residual at node $I$ splits into a displacement part and a rotation part.

---
7. Variations of strain measures

Variation of $\varepsilon_{\alpha\beta}$ w.r.t. displacement:

$$\delta\varepsilon_{\alpha\beta} = \tfrac{1}{2}(\delta a_{\alpha\beta}) = \tfrac{1}{2}(\delta\mathbf{a}\alpha \cdot
\mathbf{a}\beta + \mathbf{a}\alpha \cdot \delta\mathbf{a}\beta)$$

Since $\delta\mathbf{a}\alpha = \delta\mathbf{u}{,\alpha} = \sum_I N_{I,\alpha} \delta\mathbf{u}_I$:

$$\delta^{(I)}\varepsilon_{\alpha\beta} = \tfrac{1}{2}(N_{I,\alpha} \mathbf{a}\beta + N{I,\beta} \mathbf{a}_\alpha) \cdot
\frac{\delta\mathbf{u}_I}{|\delta\mathbf{u}_I|}$$

(as a 3-vector contribution in the $\delta\mathbf{u}I$ direction). $\varepsilon{\alpha\beta}$ does not depend on
$\boldsymbol{\phi}$.

Variation of $\kappa_{\alpha\beta}$ w.r.t. displacement (holding $\mathbf{d}$ fixed):

$$\delta_u \kappa_{\alpha\beta}^{(I)} = \tfrac{1}{2}(N_{I,\alpha} \mathbf{d}{,\beta} + N{I,\beta} \mathbf{d}_{,\alpha}) \cdot
\delta\mathbf{u}_I$$

Variation of $\kappa_{\alpha\beta}$ w.r.t. rotation at node $I$ (holding $\mathbf{a}_\alpha$ fixed):

$$\delta_\phi \kappa_{\alpha\beta}^{(I)} = \tfrac{1}{2}(\mathbf{a}\alpha \cdot \delta\mathbf{d}{,\beta} + \mathbf{a}\beta
\cdot \delta\mathbf{d}{,\alpha})$$

with $\delta\mathbf{d}{,\beta} = N{I,\beta} \frac{\partial \mathbf{d}_I}{\partial \phi_r} \delta\phi_r^I$ (variation of the
interpolated director gradient).

$$\delta_\phi \kappa_{\alpha\beta}^{(I)} = \tfrac{1}{2}(N_{I,\alpha} \mathbf{a}\beta + N{I,\beta} \mathbf{a}_\alpha) \cdot
\frac{\partial \mathbf{d}_I}{\partial \phi_r} \delta\phi_r^I$$

(using symmetry).

Variation of $\gamma_\alpha$ w.r.t. displacement:
$$\delta_u \gamma_\alpha^{(I)} = N_{I,\alpha} , \mathbf{d} \cdot \delta\mathbf{u}_I$$

Variation of $\gamma_\alpha$ w.r.t. rotation at node $I$:
$$\delta_\phi \gamma_\alpha^{(I)} = \mathbf{a}_\alpha \cdot N_I \frac{\partial \mathbf{d}_I}{\partial \phi_r} \delta\phi_r^I$$

---
8. Explicit residuals

Displacement residual (3-vector at node $I$):

Substituting the variations into $\delta W_\text{int}$ and collecting terms at $\delta\mathbf{u}_I$:

$$\mathbf{r}I^u = \int\Omega \left[ N^{\alpha\beta} \tfrac{1}{2}(N_{I,\alpha}\mathbf{a}\beta + N{I,\beta}\mathbf{a}\alpha) +
M^{\alpha\beta} \tfrac{1}{2}(N{I,\alpha}\mathbf{d}{,\beta} + N{I,\beta}\mathbf{d}{,\alpha}) + Q^\alpha N{I,\alpha} \mathbf{d}
\right] \sqrt{A} , d\Omega$$

Using the symmetry of $N^{\alpha\beta}$ and $M^{\alpha\beta}$ (both equal their transposes), each pair of terms collapses:

$$\boxed{\mathbf{r}I^u = \int\Omega N_{I,\alpha} \left[ N^{\alpha\beta} \mathbf{a}\beta + M^{\alpha\beta} \mathbf{d}{,\beta} +
Q^\alpha \mathbf{d} \right] \sqrt{A} , d\Omega}$$

(summed over $\alpha$ and $\beta$). This is a 3-vector. Define the shell stress vector:

$$\mathbf{S}^\alpha \equiv N^{\alpha\beta} \mathbf{a}\beta + M^{\alpha\beta} \mathbf{d}{,\beta} + Q^\alpha \mathbf{d}$$

Then $\mathbf{r}I^u = \int\Omega N_{I,\alpha} \mathbf{S}^\alpha \sqrt{A} , d\Omega$ — a divergence form identical in structure
to the membrane-only case, but now $\mathbf{S}^\alpha$ carries bending and shear contributions through $\mathbf{d}_{,\beta}$
and $\mathbf{d}$.

Rotation residual (scalar at node $I$, for rotation $\phi_r$, $r=1,2$):

Collecting terms at $\delta\phi_r^I$:

$$r_I^{\phi_r} = \int_\Omega \left[ M^{\alpha\beta} N_{I,\alpha} \mathbf{a}\beta + Q^\alpha N_I \mathbf{a}\alpha \right] \cdot
\frac{\partial \mathbf{d}_I}{\partial \phi_r} \sqrt{A} , d\Omega$$

Define the moment vector at the quadrature point:
$$\mathbf{m}I \equiv N{I,\alpha} M^{\alpha\beta} \mathbf{a}\beta + N_I Q^\alpha \mathbf{a}\alpha$$

Then:
$$\boxed{r_I^{\phi_r} = \int_\Omega \mathbf{m}_I \cdot \frac{\partial \mathbf{d}_I}{\partial \phi_r} \sqrt{A} , d\Omega}$$

The quantity $\partial \mathbf{d}_I / \partial \phi_r$ is the director tangent — it depends only on the Rodrigues
parametrisation at node $I$ and is derived next.

---
9. Rodrigues director tangent

The director at node $I$ (dropping node subscript):

$$\mathbf{d} = \cos\theta , \mathbf{G}_3 + \text{sinc}(\theta)(\phi_1 \mathbf{T}_1 + \phi_2 \mathbf{T}_2), \qquad \theta =
\sqrt{\phi_1^2 + \phi_2^2}$$

where $\text{sinc}(\theta) = \sin\theta/\theta$ (with $\text{sinc}(0) = 1$).

Define for brevity: $c \equiv \cos\theta$, $\sigma \equiv \sin\theta/\theta$, $\mathbf{v} \equiv \phi_1\mathbf{T}_1 +
\phi_2\mathbf{T}_2$.

Differentiating with respect to $\phi_r$ (using $\partial\theta/\partial\phi_r = \phi_r/\theta$):

$$\frac{\partial c}{\partial \phi_r} = -\sin\theta \cdot \frac{\phi_r}{\theta} = -\sigma \phi_r$$

$$\frac{\partial \sigma}{\partial \phi_r} = \frac{c - \sigma}{\theta^2} \phi_r$$

$$\frac{\partial \mathbf{d}}{\partial \phi_r} = -\sigma \phi_r , \mathbf{G}_3 + \frac{c - \sigma}{\theta^2} \phi_r ,
\mathbf{v} + \sigma , \mathbf{T}_r$$

$$\boxed{\frac{\partial \mathbf{d}}{\partial \phi_r} = \sigma , \mathbf{T}_r + \phi_r \left[ \frac{c - \sigma}{\theta^2} ,
\mathbf{v} - \sigma , \mathbf{G}_3 \right]}$$

Limit at $\theta \to 0$: Using Taylor expansions $c \to 1$, $\sigma \to 1$, $(c-\sigma)/\theta^2 \to -1/3$:
$$\left.\frac{\partial \mathbf{d}}{\partial \phi_r}\right|_{\phi=0} = \mathbf{T}_r$$

As expected: an infinitesimal rotation $\delta\phi_r$ tilts $\mathbf{G}_3$ toward $\mathbf{T}_r$.

---
10. Summary

The full RM/Naghdi residual at node $I$ is:

┌──────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────┐
│         DOF          │                                             Residual                                             │
├──────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ $\mathbf{u}_I$       │ $\mathbf{r}I^u = \int\Omega N_{I,\alpha} \left( N^{\alpha\beta} \mathbf{a}\beta +                │
│ (3-vector)           │ M^{\alpha\beta} \mathbf{d}{,\beta} + Q^\alpha \mathbf{d} \right) \sqrt{A} , d\Omega$             │
├──────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ $\phi_r^I$ (scalar,  │ $r_I^{\phi_r} = \int_\Omega \left( N_{I,\alpha} M^{\alpha\beta} \mathbf{a}\beta + N_I Q^\alpha   │
│ $r=1,2$)             │ \mathbf{a}\alpha \right) \cdot \dfrac{\partial \mathbf{d}_I}{\partial \phi_r} \sqrt{A} ,         │
│                      │ d\Omega$                                                                                         │
└──────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────┘

Inputs needed at each quadrature point — all available from ShellCellValues and the current solution:

┌───────────────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────┐
│                     Quantity                      │                               Source                                │
├───────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
│ $N_{I,\alpha}$, $N_I$                             │ scv.dNdξ, scv.N                                                     │
├───────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
│ $\mathbf{a}\alpha = \mathbf{A}\alpha +            │ scv.A₁/A₂ + $\sum_I N_{I,\alpha} \mathbf{u}_I$                      │
│ \mathbf{u}_{,\alpha}$                             │                                                                     │
├───────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
│ $\mathbf{d}$, $\mathbf{d}_{,\alpha}$              │ interpolated from nodal $\mathbf{d}_I(\boldsymbol{\phi}_I)$         │
├───────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
│ $N^{\alpha\beta}$, $M^{\alpha\beta}$, $Q^\alpha$  │ constitutive law applied to $\varepsilon_{\alpha\beta}$,            │
│                                                   │ $\kappa_{\alpha\beta}$, $\gamma_\alpha$                             │
├───────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
│ $\partial\mathbf{d}_I/\partial\phi_r$             │ Rodrigues formula above                                             │
└───────────────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────┘

Nothing here requires ForwardDiff — every term is an explicit expression in the current displacement and rotation DOFs.