```@meta
CurrentModule = FerriteShells
DocTestSetup = :(using FerriteShells)
```

# Assembly

```@docs
mass_matrix!
```

## Membrane functions (Kirchhoff-Love and Reissner-Mindlin)

```@docs
membrane_residuals!
membrane_tangent!
```

## Kirchhoff-Love functions

```@docs
bending_residuals_KL!
bending_tangent_KL!
bending_energy_KL
```

## Reissner-Mindlin functions

```@docs
residuals_and_tangent_RM!
residuals_RM!
tangent_RM!
bending_residuals_RM!
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