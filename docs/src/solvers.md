## Linear analysis

```julia
u = K \ r
```

or even better, by pre-allocating the factorization of the stiffness matrix, and then solving for different load cases

```julia
K_factor = factorize(K)
u = K_factor \ r
```

## Nonlinear analysis

### Newton–Raphson method

```julia
u = zeros(n_dofs)
residual = compute_residual(u)
while norm(residual) > tol
    tangent_stiffness = compute_tangent_stiffness(u)
    du = tangent_stiffness \ residual
    u += du
    residual = compute_residual(u)
end
```

### Load-controled Newton-Raphson


### Displacement-controled Newton-Raphson
<!-- https://doi.org/10.1016/j.compstruc.2021.106674 -->


### Arc-length method