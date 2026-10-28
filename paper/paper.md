---
title: 'FerriteShells.jl: Geometrically and materially nonlinear shell assembly in Ferrite.jl'
tags:
  - Julia
  - finite elements
  - shells
  - Ferrite.jl
authors:
  - name: Marin Lauber
    orcid: 0000-0003-2191-9318
    affiliation: "1"
  - name: Mathias Peirlinck
    orcid: 0000-0002-4948-5585
    affiliation: "1"
affiliations:
  - name: Department of Biomechanical Engineering, Faculty of Mechanical Engineering, Delft University of Technology, The Netherlands
    index: 1
date: 28 October 2026
bibliography: paper.bib
---

<!-- Items marked TODO need your input or verification before submission. Delete all comments before submitting. -->

# Summary

FerriteShells.jl is a companion library for Ferrite.jl [@FerriteJl], a Julia finite-element toolbox, that provides the building blocks to simulate and analyze structural mechanics problems involving shells. It implements a geometrically exact Reissner--Mindlin shell formulation that can be used for both thick and thin shells.

The library provides assembly functions for the energy, residual and consistent tangent, which can be combined with Ferrite.jl's degrees-of-freedom handler, global assembly and boundary conditions to construct shell problems. Static and dynamic analyses can be built from these quantities by choosing a suitable solution or time-integration scheme. The library also provides assembly functions for mass matrices, which can be used for dynamic analyses and L2-projection.

The library is compatible with most of the Ferrite.jl ecosystem, allowing users to use file I/O, distributed assembly, L2-projection and other features directly.

Some of the key features of FerriteShells.jl include:

- thick shells (Figure 1), <!-- TODO: state the t/L range you validated; the draft had t/L>5, which looks inverted. -->
- thin shells with MITC treatment to prevent shear locking (Figure 2), <!-- TODO: state the t/L range; draft had t/L<10. -->
- geometric and material nonlinearities, through a Rodrigues parametrization of the shell director and strain-energy-density-based hyperelastic material laws (Figure 3).

<!-- TODO: Figures 1-3 are not included. JOSS syntax: ![Caption](figure.png){#fig:label width=80%} and reference with @fig:label. -->

# Statement of need

Shells are structures with one dimension much smaller than the other two, which describes a wide range of engineered and biological structures, and their analysis has been the subject of numerous publications.
A kinematic assumption for the through-thickness behavior replaces the full three-dimensional description, so the structure can be described by its midsurface alone.
However, shell formulations, analysis and physical response can be complex, with wrinkling, buckling and snap-through.

FerriteShells.jl aims to bridge the gap between the formulation and the analysis of shell structures by providing assembly functions for classical shell formulations, which can readily be combined with Ferrite.jl's degrees-of-freedom handler, global assembly and boundary conditions.

Two shell formulations are commonly used in practice: Kirchhoff--Love and Reissner--Mindlin. The former is a thin-shell formulation that assumes the shell's normal remains perpendicular to the midsurface after deformation, while the latter allows for transverse shear deformation and is therefore suited to thick shells.
FerriteShells.jl implements a Reissner--Mindlin shell that can be used for both thick and thin shells, the latter being treated with the MITC method to prevent shear locking.

We note that in the literature, Reissner--Mindlin shells are sometimes referred to as geometrically exact Naghdi shells.
The main difference is that Reissner--Mindlin shells are obtained through a specialization of three-dimensional elasticity to the shell's midsurface, while Naghdi shells start from the midsurface and postulate the existence of a Cosserat-type surface normal.
The final expressions for the membrane, bending and shear strain measures are identical in the two formulations.

# State of the field

Shell formulations tightly couple the mathematical formulation and the software implementation, because the local surface metric required to evaluate the structure's internal energy is not needed in classical three-dimensional elasticity.
This complicates the implementation of shells in existing finite-element frameworks.
Additionally, shells are prone to membrane and shear locking, which require additional treatment that often depends on the type of structure under analysis.

General-purpose open-source solvers such as CalculiX [@CalculiX] and MoFEM [@MoFEM] support shells through 3D continuum elements: in CalculiX, for example, the shell midsurface is expanded internally into 3D wedge elements, whereas MoFEM implements a hierarchical, heterogeneous, and anisotropic prism finite element designed for solid shells.
These elements are prone to the same locking phenomena as true shells and require reduced integration, MITC or assumed-strain treatments, but they allow the use of 3D hyperelastic material laws.

More general shell formulations, based on weak forms and automatic code generation, are available through the FEniCS ecosystem [@FEniCSShells]. FEniCS-shells is written in Python and relies on a different formulation, whereas FerriteShells.jl targets Julia users and a geometrically exact Reissner--Mindlin formulation with hyperelastic laws specified through automatic differentiation. <!-- TODO: say concretely which formulation FEniCS-shells uses and why it does not fit your use case. -->
Within Julia, general finite-element toolboxes such as Gridap [@Gridap] and Ferrite.jl provide the building blocks for solid mechanics, but, to our knowledge, Ferrite.jl has no full support for geometrically and materially nonlinear shells. <!-- TODO: state the Ferrite.jl version checked and link the relevant issue/discussion; add one sentence on Gridap shell support. Consider a sentence on Kratos/deal.II/Code_Aster. -->
Because shells are assembled differently from solids, requiring surface metrics, a director field and rotational degrees of freedom, we implement them as a companion package rather than within Ferrite.jl itself.

Lastly, shells are often used to model structures with nonlinear anisotropic material behavior, such as arteries, hearts or soft robotic actuators. In these settings, hyperelastic formulations are commonly used, in which a strain-energy density function is expressed in terms of the (3D) right Cauchy--Green deformation tensor in the Cartesian frame.
In this case, the three-dimensional elasticity must be specialized onto the curvilinear coordinate system attached to the shell.

# Software design

The overall idea of FerriteShells.jl is to adapt Ferrite.jl's `CellValues` and the associated `reinit!` to shells, so that the package integrates with the whole Ferrite.jl ecosystem. Since shells rely on specializing the governing equations to the curvilinear system associated with the structure's midsurface, the new `ShellCellValues` holds classical surface quantities such as the metric tensor and local curvilinear coordinates. `ShellCellValues` is then used within assembly functions for the energy, residual and consistent tangent. This enables integration into the Ferrite.jl environment with minimal code changes compared to native Ferrite.jl problems, and allows users to become proficient quickly in assembling and solving shell problems.

```julia
scv = ShellCellValues(qr, ip_geo, ip_shape; mitc)
for cell in CellIterator(dh)
    reinit!(scv, cell)
    u_e = u[shelldofs(cell)]
    residuals_RM!(re, scv, u_e, mat)
    tangent_RM!(ke, scv, u_e, mat)
    assemble!(assembler, shelldofs(cell), ke, re)
end
```

`ShellCellValues` inherits the geometric interpolation (`ip_geo::Interpolation`), the function interpolation (`ip_shape::Interpolation`) and the quadrature rule (`qr::QuadratureRule`) from Ferrite, allowing structures to be modeled with linear and quadratic triangular and quadrilateral elements. <!-- TODO: confirm supported element types (draft: "linear and quadratic triangular and quadratic elements"). Also check the Ferrite type name for quadrature rules. -->

FerriteShells.jl relies on Julia's multiple dispatch [@Julia] to specialize Ferrite.jl functions to `ShellCellValues` and to select the correct behavior in its internal assembly functions.

FerriteShells.jl also provides a helper function, `shell_grid`, that embeds a two-dimensional Ferrite `Grid{2}` in a three-dimensional `Grid{3}` with the same topology, so that users can easily define shell problems from 2D meshes.

```julia
corners = [Vec{2}(( 0.0,  0.0)), Vec{2}((48.0, 44.0)),
           Vec{2}((48.0, 60.0)), Vec{2}(( 0.0, 44.0))]
grid = shell_grid(generate_grid(QuadraticQuadrilateral, (32, 32), corners))
```

Compatibility with most of Ferrite's ecosystem is preserved for `ShellCellValues`, allowing users to use file I/O, distributed assembly, L2-projection and other Ferrite features. <!-- TODO: finish this sentence from the draft (it was cut off) and say which features were actually tested. -->

Multiple dispatch is also used to specialize the shear treatment to the different methods available in the literature. The MITC method [@BatheDvorkin1985] is implemented as an abstract type `AbstractMITC`, with concrete types for the individual variants (e.g. `MITC3`, `MITC6`, `MITC9`) selected by dispatch. <!-- TODO: the draft listed MITC3 twice; confirm the exact variants (MITC4?). --> Users can switch between MITC methods by changing the type passed to `ShellCellValues`, and the design allows other shear treatments, such as selective reduced integration, to be added in the future.

FerriteShells.jl leverages Julia's automatic differentiation to model strain-energy-based hyperelastic materials through a simple interface: the (scalar) strain-energy function is specified in terms of the Cartesian right Cauchy--Green deformation tensor ($C = F^\top F$), and is internally specialized to the curvilinear coordinate system of the shell through an incompressibility assumption.

```julia
material = Hyperelastic(C -> μ/2 * (tr(C) - 3), thickness)
```

This spares users modeling hyperelastic shells from specializing the material model to the shell's coordinate system themselves.
<!-- TODO: say whether anisotropy (fiber directions) and compressible laws are supported. The motivation mentions anisotropic tissue, but the only example is isotropic. -->
<!-- TODO: add a sentence on design trade-offs (why ShellCellValues, AD for materials vs. analytic shell kinematics) and on performance. -->

# Limitations

<!-- TODO: list what is not supported, e.g. incompressible-only material reduction, no contact, no shell-solid coupling, drilling rotation treatment, mass matrix, etc. -->

# Research impact statement

<!-- TODO: replace with evidence rather than expected impact. Suggested content:
- Validation benchmarks (e.g. Scordelis-Lo roof, pinched cylinder/hemisphere, hyperelastic inflation) with a convergence plot.
- Use in the Holland Hybrid Heart project and in manuscripts in preparation (preprint links if available).
- Test suite, documentation, registry status (registered in the Julia General registry?), external users, issues.
-->

FerriteShells.jl was developed within the Holland Hybrid Heart project, where it is used to model thin, hyperelastic membranes and shells. It is also used in two manuscripts currently in preparation. <!-- TODO: verify wording and add references/preprints. -->

# AI usage disclosure

Most of the software in this library was developed with the help of agentic AI (Claude Opus 4.8) under close supervision of the first author. The software architecture, the binding with Ferrite.jl, and the general implementation route were defined by the first author. <!-- TODO: confirm exact model/tool names and versions; state which parts were AI-assisted (code, tests, documentation, paper text) and how the output was verified (tests, benchmarks, manual review). -->

# Acknowledgements

This software was developed as part of the Holland Hybrid Heart project with file number NWA.1518.22.049 of the research program Onderzoek op Routes door Consortia 2022 -- NWA-ORC 2022, which is financed by the Dutch Research Council (NWO), the Dutch Ministry of Education, Culture and Science (OCW), and the Hartstichting (Dutch Heart Foundation).
<!-- TODO: add contributors/collaborators to thank, or leave this out. -->

# References