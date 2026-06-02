```@meta
CurrentModule = FerriteShells
DocTestSetup = :(using FerriteShells)
```

# Assembly

```@docs
mass_matrix!
```

## Kirchhoff-Love functions

```@docs
membrane_residuals_KL!
membrane_tangent_KL!
bending_residuals_KL!
bending_tangent_KL!
bending_energy_KL
```

## Reissner-Mindlin functions

```@docs
membrane_residuals_RM!
bending_residuals_RM!
membrane_tangent_RM!
bending_tangent_RM!
residuals_RM_FD!
tangent_RM_FD!
energy_RM
```

## External loading functions

```@docs
assemble_traction!
apply_pointload!
assemble_pressure!
assemble_pressure_tangent!
```

## Volume functions

```@docs
volume_residuals!
volume_gradient!
```