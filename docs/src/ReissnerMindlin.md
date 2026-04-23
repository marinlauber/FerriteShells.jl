# Reissner-Mindlin / Naghdi shell

```@raw html
<img src="./images/shell_kinematic.png" style="background-color:white;" />
```

## 1. Kinematics

The Reissner-Mindlin kinematic relaxes the kirchhoff-Love zero shear strain assumption through orthogonality of material lines. The shear strain measures the rotation of these material lines around the normal vector of the shell's midsurface ``\hat{\mathbf{a}}_3``
```math
\Phi(\xi^1,\xi^2,\xi^3) = \phi(\xi^1,\xi^2) + \xi^3\theta^\lambda(\xi^1,\xi^2) \mathbf{a}_\lambda(\xi^1,\xi^2) = \phi(\xi^1,\xi^2) + \xi^3\mathbf{d}(\xi^1,\xi^2)
```
where ``\mathbf{d}(\xi^1,\xi^2)`` is the director at a point ``(\xi^1,\xi^2)`` on the midsurface and ``\gamma_\alpha=\mathbf{d}⋅ \mathbf{a}_\alpha``.

The surface basis vector are given by
```math
\begin{split}
\mathbf{g}_\alpha &= \frac{\partial \Phi(\xi^1,\xi^2)}{\partial \xi^\alpha} = \frac{\partial}{\partial\xi^\alpha}\left[\phi(\xi^1,\xi^2) + \xi^3\mathbf{d}_3(\xi^1,\xi^2)\right]\\
&= \mathbf{a}_\alpha + \xi^3 \mathbf{d}_{,\alpha} \\
\mathbf{g}_3 &= \frac{\partial}{\partial\xi^3}\left[\phi(\xi^1,\xi^2) + \xi^3\mathbf{d}(\xi^1,\xi^2)\right] = \mathbf{d}
\end{split}
```
using the definition of ``\mathbf{a}_\alpha``. From this, we can get the components of the metric tensor
```math
\begin{split}
g_{\alpha\beta} &= \mathbf{g}_\alpha \cdot \mathbf{g}_\beta = (\mathbf{a}_\alpha+\xi^3\mathbf{d}_{,\alpha})\cdot(\mathbf{a}_\beta+\xi^3\mathbf{d}_{,\beta})\\
 &= a_{\alpha\beta} + \xi^3(\mathbf{a}_\alpha\cdot\mathbf{d}_{,\beta} + \mathbf{a}_\beta\cdot\mathbf{d}_{,\alpha}) + (\xi^3)^2\mathbf{d}_{,\alpha}\cdot\mathbf{d}_{,\beta}\\
 &= a_{\alpha\beta} + \xi^3(\mathbf{a}_\alpha\cdot\mathbf{d}_{,\beta} + \mathbf{a}_\beta\cdot\mathbf{d}_{,\alpha}) + O(t^2) \\
g_{\alpha 3} &= g_{3\alpha} = (\mathbf{a}_\alpha+\xi^3\mathbf{d}_{,\alpha})\cdot\mathbf{d} = \mathbf{a}_\alpha\cdot\mathbf{d} + \xi^3\mathbf{d}_{,\alpha}\cdot\mathbf{d}\\
g_{33} &= 1.
\end{split}
```
where the plane stress assumption results in ``g_{33}=1`` and the shear contributions are non-zero ``g_{3\alpha}\neq0``. As a result, the strain tensor is now
```math
\begin{split}
e_{\alpha\beta} &= \frac{1}{2}(g_{\alpha\beta} - G_{\alpha\beta})\\
e_{\alpha3} &= \\
e_{33} &= 0.
\end{split}
```

!!! note
    Interestingly, the plane stress assumption now results in non-zero transverse strains ``e_{3\alpha}\neq0``. This is expected since shear also results in ... This component scales with the thickness squared and is usually very small. We can recover it as a post-processing step from the in-plane strain via
    ```math
    e_{3\alpha} = ...
    ```

### 1.1 Director parametrization

There are a few ways to parametrize the the director vector, and the different choice lead to different discertization. One way is to discretize each of its components, leading to an additional 3 degrees of freedom per node. This is the simplest way, but requires enforcing ``\Vert\mathbf{d}\Vert=1`` through a Lagrange multiplier approach and static condensation, which results in an overall complex implementation.

Another way is to use additive vector rotations starting from the midsurface normal
```math
\mathbf{d} = \hat{\mathbf{a}}_3 + \theta_1\mathbf{T}_1 + \theta_2\mathbf{T}_2
```
which removes one unknown since we only require ``\theta_1,\theta_2`` to fully describe ``\mathbf{d}``. One issue with this formulation is that the unitarity of the director is not enforced ``\Vert\mathbf{d}\Vert\neq1``. This limits the formulation to small rotations ``\Vert\mathbf{\theta}\Vert\ll1`` as large ``\Vert\mathcal{d}\Vert`` would lead to large shear strains (``\gamma_\alpha=\mathbf{a}_\alpha\cdot\mathbf{d}``) resulting in shear locking as all the internal energy is taken by shear.

For finite rotation nonlinear shell, we would like to parametrize ``\mathbf{d}`` in a way that naturally enforces the ``\Vert\mathbf{d}\Vert=1`` constraint. One way to do this is through Rodrigue's parametrization
```math
\mathbf{d} = \cos{\Vert\mathbf{\theta}\Vert}\cdot\hat{\mathbf{a}}_3 + \text{sinc}{\Vert\theta\Vert}\cdot(\theta_1\cdot\mathbf{T}_1 + \theta_2\cdot\mathbf{T}_2)
```
which guarantees ``\Vert\mathcal{d}\Vert=1`` for rotations that satisfy ``\mathbf{\theta}^2 = \theta_1^2 + \theta_2^2`` . This formulation is also limited by a singularity in the Rodrigue parametrization for ``\theta=\pi`` rotations. This could be solved with quarterion parametrization, but in practice, an updated Lagrange formation can be used to enforce ``\theta<\phi``.
In the following, we will keep the director variation terms general since explicit variation of the director is messy, especially here since we use a Rodrigue's parametrization.

!!! info
    In practice, we use ``\theta^2`` in the trigonometric functions to enforce directly the constraint on the rotations, but this means that for small rotations, we could take the square-root of a very small number, which could lead to overflow. To avoid this, we use a Taylor-series expansion to evaluate the trigonometric functions for ``\mathbf{\theta}^2<10^{-6}``, and the normal expression otherwise.

## 2. Internal energy

### 2.1 Residual and first variation

### 2.2 Consistent tangent and second variation