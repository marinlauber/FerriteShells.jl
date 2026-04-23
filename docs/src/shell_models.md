# Shell theory comparison

| | Linear KL | Koiter | Reissner–Mindlin | Naghdi |
|---|---|---|---|---|
| **DOFs/node** | 3 (``u_1,u_2,u_3``) | 3 (``u_1,u_2,u_3``) | 5 (``u_1,u_2,u_3,\varphi_1,\varphi_2``) | 5 (``u_1,u_2,u_3,\varphi_1,\varphi_2``) |
| **Director** | implicit: ``\mathbf{n} = \mathbf{a}_1\times\mathbf{a}_2/\|\cdot\|`` | implicit: ``\mathbf{n} = \mathbf{a}_1\times\mathbf{a}_2/\|\cdot\|`` | additive: ``\mathbf{d} = \mathbf{G}_3+\varphi_1\mathbf{T}_1+\varphi_2\mathbf{T}_2``, ``\|\mathbf{d}\|\neq 1`` | Rodrigues: ``\mathbf{d} = \cos\|\varphi\|\,\mathbf{G}_3+\mathrm{sinc}\|\varphi\|(\varphi_1\mathbf{T}_1+\varphi_2\mathbf{T}_2)``, ``\|\mathbf{d}\|=1`` |
| **Membrane strain** | linear: ``\tfrac{1}{2}(\mathbf{A}_\alpha\cdot\mathbf{u}_{,\beta}+\mathbf{A}_\beta\cdot\mathbf{u}_{,\alpha})`` | Green–Lagrange: ``\tfrac{1}{2}(a_{\alpha\beta}-A_{\alpha\beta})`` | linear: ``\tfrac{1}{2}(\mathbf{A}_\alpha\cdot\mathbf{u}_{,\beta}+\mathbf{A}_\beta\cdot\mathbf{u}_{,\alpha})`` | Green–Lagrange: ``\tfrac{1}{2}(a_{\alpha\beta}-A_{\alpha\beta})`` |
| **Bending strain** | ``\kappa_{\alpha\beta} = -u_{3,\alpha\beta}`` (flat ref.) | ``\kappa_{\alpha\beta} = b_{\alpha\beta}-B_{\alpha\beta}`` | ``\tfrac{1}{2}(\mathbf{A}_\alpha\cdot\mathbf{d}_{,\beta}+\mathbf{A}_\beta\cdot\mathbf{d}_{,\alpha})-B_{\alpha\beta}`` | ``\tfrac{1}{2}(\mathbf{a}_\alpha\cdot\mathbf{d}_{,\beta}+\mathbf{a}_\beta\cdot\mathbf{d}_{,\alpha})-B_{\alpha\beta}`` |
| **Transverse shear** | ``\gamma=0`` (Kirchhoff) | ``\gamma=0`` (Kirchhoff) | ``\gamma_\alpha = \mathbf{A}_\alpha\cdot\mathbf{d}`` | ``\gamma_\alpha = \mathbf{a}_\alpha\cdot\mathbf{d}`` |
| **Finite rotations** | no | yes | small only (``\|\varphi\|\ll 1``) | yes |
| **C¹ for bending** | yes | yes | no | no |
| **In FerriteShells** | — | ✓ `_KL` functions | — | ✓ `_RM` functions |

The key distinction between RM and Naghdi is which base vectors appear in the strain measures: RM uses the **reference** base vectors ``A_\alpha`` (linearised around the reference configuration), while Naghdi replaces them with the **current** ``a_\alpha`` everywhere, giving fully nonlinear strains. The director parametrisation (non-unit additive vs unit Rodrigues) is a separate but related choice — in practice the two always appear together.

Koiter has no director DOFs; the normal is always implicit from the surface geometry, so the Kirchhoff constraint (zero shear) is built in and C¹ continuity is required for bending.

Classical RM (additive director, ``\|\mathbf{d}\|\neq 1``) is not implemented; the `_RM` functions go directly to the geometrically exact Naghdi form via Rodrigues parametrisation.