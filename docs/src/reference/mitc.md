```@meta
CurrentModule = FerriteShells
DocTestSetup = :(using FerriteShells)
```

More concretely, when you sample a quadratic shear strain (which has a spurious locking term) at these two points and then interpolate linearly between them, the quadratic parasitic term averages out exactly. That is the mechanism of locking removal.

For Q4 (bilinear displacement), the shear strain polynomial spaces follow directly from differentiating the displacement field w = a₀ + a₁ξ + a₂η + a₃ξη:
∂w/∂ξ = a₁ + a₃η   →  constant in ξ, linear in η
∂w/∂η = a₂ + a₃ξ   →  linear in ξ, constant in η

MITC Type

```@docs
MITC
MITC9
MITC4
```

```@docs
tying_shear_strains
shear_strains
```