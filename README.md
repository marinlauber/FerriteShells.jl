[![Test](https://github.com/marinlauber/FerriteShells.jl/actions/workflows/test.yml/badge.svg)](https://github.com/marinlauber/FerriteShells.jl/actions/workflows/test.yml)
[![codecov.io](https://codecov.io/github/marinlauber/FerriteShells.jl/coverage.svg?branch=master)](https://codecov.io/github/marinlauber/FerriteShells.jl?branch=master)
[![][docs-stable-img]][docs-stable-url]

# FerriteShells.jl

---

This package provides helper functions to assemble the different terms in the weak form of most classical shell formulations — C⁰ Kirchhoff–Love linear, C⁰ Koiter (non-linear Kirchhoff–Love), Reissner–Mindlin, and Naghi (non-linear Reissner–Mindlin) shells.
Specifically, this package provides functions to assemble the classical membrane, bending, and shear contributions to the residuals and the consistent tangent stiffness matrix.
Proper assembly of these different terms leads to the different formulation mentioned above.

We refer the reader to the specific weak form for each of these shells and their numerical implementation limitations.

Some formulation that can be assembled with this package

Function | Membrane | Kirchhoff–Love | Reissner–Mindlin
:------------ | :-------------| :-------------| :-------------
linear | :white_check_mark: |  :white_check_mark: | :white_check_mark:
non-linear | :white_check_mark: |  :white_check_mark: | :white_check_mark:
`Lagrange{RefTriangle, 1}` (Q3) | :white_check_mark: |  :x: | :white_check_mark:
`Lagrange{RefQuadrilateral, 1}` (Q4) | :white_check_mark: |  :x: | :white_check_mark:
`Lagrange{RefTriangle, 2}` (Q6) | :white_check_mark: |  :ballot_box_with_check: | :white_check_mark:
`Serendipity{RefQuadrilateral, 2}` (Q8) | :white_check_mark: |  :ballot_box_with_check: | :white_check_mark:
`Lagrange{RefQuadrilateral, 2}` (Q9) | :white_check_mark: |  :ballot_box_with_check: | :white_check_mark:
MITC |  |   | :construction_worker:

> [!WARNING]
> Kirchhoff–Love shells with C⁰ continuity between elements is fundamentally wrong; it works in some cases with small deformations and specific boundary conditions. I would suggest using the Reissner–Mindlin shell instead.

### `ShellCellValues`

Since most weak forms used in shell analysis are specified by specializing the classical weak form to the curvilinear system of the mid-plane of the shell, classical continuum mechanics quantities, such as the deformation gradient tensor $\bf{F}$, change when expressed in curvilinear coordinates.
To help assemble these specific quantities, this package provides a new `ShellCellValues<:AbstractCellValues`, which behaves identically to Ferrite's `CellValues`, but holds covariant basis vectors, metric tensors, and surface Jacobian at the integration points, which are used in the assembly of the different terms in the different formulations.

```julia
struct ShellCellValues <: AbstractCellValues
    ...
end
```

### Kinematics

```julia
for qp in 1:getnquadpoints(scv)
    a₁, a₂, A_metric, a_metric = kinematics(scv, qp, u_e)
    E = 0.5 * (a_metric - A_metric) # half the increment in the metric tensor
    C = contravariant_elasticity(mat, A_metric)
    N = C ⊡ E
end
```

### Shell obstacle course

#### Cook's membrane

![Cook's membrane](/docs/src/images/cooks_membrane.png)

#### Scordelis-Lo roof

![Scordelis-Lo roof](/docs/src/images/scoreldis_lo_roof.png)

#### Pinched cylinder

![Pinched cylinder](/docs/src/images/pinched_cylinder.png)

#### Cantilever roll-up

![Cantilever roll-up](/docs/src/images/cantilever_rollup.png)

#### Hyperbolic paraboloid

![Hyperbolic paraboloid](/docs/src/images/hyperbolic_paraboloid.png)

#### Square airbag

![Square airbag](/docs/src/images/airbag.png)

### Authors

- Marin Lauber, Delft University of Technology, The Netherlands.

### Contributing

We are always looking for contributions and help with FerriteShells. If you
have ideas, nice applications, or code contributions, then we would be happy to
help you get them included. We ask you to follow the FerriteShells git
workflow.

### Issues and Support

Please use the GitHub issue tracker to report any issues.

### License

FerriteShells is released under the MIT License. See the [LICENSE](LICENSE) file for details.

[docs-stable-img]: https://img.shields.io/badge/docs-latest%20release-blue
[docs-stable-url]: https://FerriteShells.github.io/FerriteShells.jl/
