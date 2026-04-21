[![Test](https://github.com/marinlauber/FerriteShells.jl/actions/workflows/test.yml/badge.svg)](https://github.com/marinlauber/FerriteShells.jl/actions/workflows/test.yml)
[![codecov.io](https://codecov.io/github/marinlauber/FerriteShells.jl/coverage.svg?branch=master)](https://codecov.io/github/marinlauber/FerriteShells.jl?branch=master)
[![][docs-stable-img]][docs-stable-url]

# FerriteShells.jl

> [!WARNING]
> This package is under active development, as such nothing is guaranteed and breaking changes might occur.

This package provides helper functions to assemble the different terms in the weak form of most classical shell formulations — C⁰ Kirchhoff–Love linear, C⁰ Koiter (non-linear Kirchhoff–Love), Reissner–Mindlin, and Naghi (non-linear Reissner–Mindlin) shells.
Specifically, the classical membrane, bending, and shear contributions to the residuals and the consistent tangent stiffness matrix can be integrated and used with [Ferrite.jl](https://ferrite-fem.github.io/Ferrite.jl/stable/).

> [!NOTE]
> This package assumes that the shell is defined by a 2D mesh embeded in 3D space `Grid{3, P, T}` where `P<:Union{Triangle, Quadrilateral, QuadraticTriangle, QuadraticQuadrilateral}`. To embed Ferrte's `generate_grid` into 3D space, we provide a simple helper function `shell_grid(grid::Grid{2, P, T}; map) -> Grid{3, P, T}` , where the `map` can be used to map the 2D grid into 3D space.

Some formulation that can be assembled with this package:

Function | Membrane | Kirchhoff–Love | Reissner–Mindlin
:------------ | :-------------| :-------------| :-------------
linear | :white_check_mark: |  :white_check_mark: | :white_check_mark:
non-linear | :white_check_mark: |  :white_check_mark: | :white_check_mark:
`Lagrange{RefTriangle, 1}` (T3) | :white_check_mark: |  :x: | :white_check_mark:
`Lagrange{RefQuadrilateral, 1}` (Q4) | :white_check_mark: |  :x: | :white_check_mark:
`Lagrange{RefTriangle, 2}` (T6) | :white_check_mark: |  :ballot_box_with_check: | :white_check_mark:
`Serendipity{RefQuadrilateral, 2}` (Q8) | :white_check_mark: |  :ballot_box_with_check: | :white_check_mark:
`Lagrange{RefQuadrilateral, 2}` (Q9) | :white_check_mark: |  :ballot_box_with_check: | :white_check_mark:
MITC |  |   | :construction_worker:

We refer the reader to the documentation for the specific weak form, numerical implementation and limitation of the different shell models.

> [!WARNING]
> Kirchhoff–Love shells with C⁰ continuity between elements is fundamentally wrong; it works in some cases with small deformations and specific boundary conditions. I would suggest using the Reissner–Mindlin shell instead.

### `ShellCellValues`

Shells specialize the classical weak form obtained in continuum mechanics to a curvilinear coordinate system located on the shell's midsurface. As a result, classical continuum mechanics quantities, such as the Green–Lagrange strain tensor $\bf{E}$ of the elasticity tensor $\mathbb{C}$, change.

To help assemble these specific surface metrics, this package uses a new `ShellCellValues<:AbstractCellValues`, which behaves identically to Ferrite's `CellValues`, but additionally holds covariant basis vectors, metric tensors, and surface Jacobian at the integration points, which are used in the assembly of the different terms of the different formulations.

```julia
struct ShellCellValues{QR, IPG, IPS, T<:AbstractFloat, M} <: AbstractCellValues
    # quadrature and interpolation spaces
    qr       :: QR
    ip_geo   :: IPG
    ip_shape :: IPS
    # same as CellValues
    N, dNdξ, d2Ndξ2, detJdV :: Various{T}
    # additional fields for shells
    A₁, A₂, A₁₁, A₁₂, A₂₂ :: Vector{Vec{3, T}}
    G₃, T₁, T₂            :: Vector{Vec{3, T}}
    # shell measures
    A_metric :: Vector{SymmetricTensor{2, 2, T, 3}}
    B        :: Vector{SymmetricTensor{2, 2, T, 3}}
    # shear-locking treatment
    mitc     :: AbstractMITC
end
```

Calling `reinit!(scv::ShellCellValues)` computes the fixed covariant basis vectors $\bf{A}_1$ and $\bf{A}_2$ from the geometry of the shell's midsurface, while the current covariant basis vectors $\bf{a}_1$ and $\bf{a}_2$ are computed from the current configuration of the shell. The metric tensors $\bf{A}_{\alpha\beta}$ and $\bf{a}_{\alpha\beta}$ are then obtained as the inner product of the corresponding covariant basis vectors.

From these surface measures and the contravariant elasticity tensor $\mathbb{C}^{\alpha\beta\gamma\delta}$, the membrane, bending and shear strains can be computed, which are used in the assembly of the different terms in the different formulations.

### Global assembly

Assembling the element contributions into the global system is identical to Ferrite, but instead of calling `CellValues`, the user needs to call `ShellCellValues` and use the corresponding assembly functions for the different terms in the different formulations. For example, for a non-linear Reissner–Mindlin shell, the assembly of the global consistent stiffness matrix and residual vector can be done as follows:

```julia
function assemble_shell!(K_int, r_int, dh, scv, u, mat)
    n_e = ndofs_per_cell(dh)
    ke_i = zeros(n_e, n_e); re_i = zeros(n_e)
    asm_i = start_assemble(K_int, r_int)
    for cell in CellIterator(dh)
        fill!(ke_i, 0.0); fill!(re_i, 0.0)
        reinit!(scv, cell)
        sd  = shelldofs(cell)
        u_e = u[sd]
        membrane_residuals_RM!(re_i, scv, u_e, mat)
        bending_residuals_RM!(re_i, scv, u_e, mat)
        membrane_tangent_RM!(ke_i, scv, u_e, mat)
        bending_tangent_RM!(ke_i, scv, u_e, mat)
        assemble!(asm_i, sd, ke_i, re_i)
    end
end
```

where `shelldofs` is a helper function (similar to `celldofs`) to get the degrees of freedom of the shell element, which are ordered as follows: first the in-plane displacements, then the out-of-plane displacements, and finally the rotations.

> [!WARNING]
> `shelldofs` is only usefull for Reissner–Mindlin shells where both displacements and rotations are degrees of freedom. For Kirchhoff–Love shells, the degrees of freedom are only the displacements, and the rotations are obtained from the displacements. In this case, `celldofs` must be used instead of `shelldofs`.

### External loadings

The package also provides helper functions to assemble external loading contributions, such as follower pressure loads or edge tractions, which are often used in shell problems. For terms that depend on the current configuration of the shell (i.e. follower pressure loads) the contribution of these loadings is included in the consistent tangent stiffness matrix, which is necessary for quadratic convergence of the non-linear solver.

Loading | residual | consistent tangent
:------------ | :-------------| :-------------
follower pressure | `assemble_pressure!` | `assemble_pressure_tangent!`
edge traction | `assemble_traction!` |
point load | `apply_pointload!` |
volume coupling | `volume_residuals!` | `volume_gradient!`

## Examples

Below are some examples of shell problems assembled using this package.

### Pillow Inflation

Dynamic inflation of a square pillow made of a thin Reissner-Mindlin shell (MITC9 treatment) under a uniform pressure loading.

![Square airbag](/docs/src/images/pillow.gif)

### Membrane Shear

Wrinkling of a Reissner-Mindlin thin shell with MITC4 treatment under pure lateral shear. Non-linear Newton-Raphson solver with Pseudo-Transient Continuation (PTC) to traverse the unstable equilibrium path.

![Membrane Shear](/docs/src/images/shear.gif)

### Authors

- [Marin Lauber](https://marinlauber.github.io/), Delft University of Technology, The Netherlands.

### Contributing

We are always looking for contributions and help with FerriteShells. If you
have ideas, nice applications, or code contributions, then we would be happy to
help you get them included. We ask you to follow the FerriteShells git
workflow.

### Issues and Support

Please use the GitHub issue tracker to report any issues.

### License

FerriteShells is released under the MIT License. See the [LICENSE](LICENSE) file for details.

[docs-stable-img]: https://img.shields.io/badge/docs-dev-blue
[docs-stable-url]: https://marinlauber.github.io/FerriteShells.jl/dev/
