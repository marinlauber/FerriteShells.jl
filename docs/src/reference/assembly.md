```@meta
CurrentModule = FerriteShells
DocTestSetup = :(using FerriteShells)
```

# Assembly

## Kirchhof-Love functions

```@docs
membrane_residuals_KL!
membrane_tangent_KL!
bending_energy_KL
```

## Reissner-Mindlin functions

```@docs
membrane_residuals_RM!
bending_residuals_RM!
membrane_residuals_RM_FD!
bending_residuals_RM_FD!
bending_shear_energy_RM
membrane_energy_RM
membrane_tangent_RM!
bending_tangent_RM!
membrane_tangent_RM_FD!
bending_tangent_RM_FD!
```

## External loading functions

```@docs
assemble_traction!
apply_pointload!
assemble_pressure_tangent!
```