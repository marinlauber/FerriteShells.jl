```@meta
CurrentModule = FerriteShells
DocTestSetup = :(using FerriteShells)
```

More concretely, when you sample a quadratic shear strain (which has a spurious locking term) at these two points and then interpolate linearly between them, the quadratic parasitic term averages out exactly. That is the mechanism of locking removal.

For Q4 (bilinear displacement), the shear strain polynomial spaces follow directly from differentiating the displacement field ``w = a_0 + a_1\\xi + a_2\\eta + a_3\\xi\\eta`` with respect to the parametric coordinates;
``\\partial w/\\partial\\xi = a_1 + a_3\\eta`` which is constant in ``\\xi``, linear in ``\\eta`` and similarly for ``\\partial w/\\partial\\eta = a_2 + a_3\\xi`` which is linear in ``\\xi``, constant in ``\\eta``

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