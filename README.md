[![Test](https://github.com/marinlauber/FerriteShells.jl/actions/workflows/test.yml/badge.svg)](https://github.com/marinlauber/FerriteShells.jl/actions/workflows/test.yml)

# FerriteShells.jl

Assemblers for shells in Ferrite.jl

This package provides helper functions to assemble the different terms in the weak form of most classical shell formulation — C⁰ Kirchhoff--Love linear, C⁰ Koiter (non-linear Kirchhoff-Love), Reissner-Mindlin and Naghi (non-linear Reissner-Mindlin) shells.
Specifically, this package provides helper function to assemble the classical membrane, bending and shear contribution to the residuals and the consistent tangent stiffness matrix.
Proper assembly of these different terms lead to the different formulation mentioned above.

We refer to the reader to the specific weak form for each of these shell and their numerical an implementation limitations.

Some formulation that can be assembled with this package

Function | Membrane | Kirchhoff-Love | Reissner-Mindlin
:------------ | :-------------| :-------------| :-------------
linear | :white_check_mark: |  :white_check_mark: | :white_check_mark:
non-linear | :white_check_mark: |  :white_check_mark: | :white_check_mark:
`Lagrange{RefTriangle, 1}` (Q3) | :white_check_mark: |  :x: | :white_check_mark:
`Lagrange{RefQuadrilateral, 1}` (Q4) | :white_check_mark: |  :x: | :white_check_mark:
`Lagrange{RefTriangle, 2}` (Q6) | :white_check_mark: |  :ballot_box_with_check: | :white_check_mark:
`Serendipity{RefQuadrilateral, 2}` (Q8) | :white_check_mark: |  :ballot_box_with_check: | :white_check_mark:
`Lagrange{RefQuadrilateral, 2}` (Q9) | :white_check_mark: |  :ballot_box_with_check: | :white_check_mark:
MITC |  |   | :construction_worker:

## 1. `ShellCellValues`

Since most weak forms use in shell analysis are specified by specializing the classical weak form to the curvilinear system of the mid-plane of the shell, classical continuum mechanics quantities, such as the deformation gradient tensor $\bf{F}$ change when expressed in curvilinear coordinates.
To help assemble these specific quantities, this package provides a new `ShellCellValues<:AbstractCellValues`, which behaves identically to Ferrite's `CellValues`, but hold covariant basis vector, metric tensors and surface Jacobian at the integration points.
```julia
struct ShellCellValues <: AbstractCellValues
    ...
end
```

## 2. Kinemtiics

```julia
for qp in 1:getnquadpoints(scv)
    a₁, a₂, A_metric, a_metric = kinematics(scv, qp, u_e)
    E = 0.5 * (a_metric - A_metric) # half the increment in the metric tensor
    C = contravariant_elasticity(mat, A_metric)
    N = C ⊡ E
end
```

> [!WARNING]
> For now, the residual and consistent tangent construction use `ForwardDiff.gradient` and `ForwardDiff.hessian`, so simplify implementation. This will be slow on large meshes (>1000 elements), in the future explicit expressions should replace those. Medium term, we can start with the membrane term only for the Reissner-Mindlin shell.

## Authors

- Marin Lauber, Delft University of Technology, The Netherlands.

## Contributing

We are always looking for contributions and help with FerriteShells. If you
have ideas, nice applications or code contributions then we would be happy to
help you get them included. We ask you to follow the FerriteShells git
workflow.

## Issues and Support

Please use the GitHub issue tracker to report any issues.

## License

FerriteShells is released under the MIT License. See the [LICENSE](LICENSE) file for details.