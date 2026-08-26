---
title: 'FerriteShells.jl: Geometrical and material non-linear shell assembly in Ferrite.jl'
tags:
  - Julia
  - finite elements
  - shell elements
  - Ferrite.jl
authors:
  - name: Marin Lauber
    orcid: 0000-0003-2191-9318
    affiliation: "1"
  - name: Viola Bini
    orcid: 0009-0000-2522-827X
    affiliation: "1"
  - name: Mathias Peirlinck
    orcid: 0000-0002-4948-5585
    affiliation: "1"
affiliations:
  - name: Department of Biomechanical Engineering, Faculty of Mechanical Engineering, Delft University of Technology, The Netherlands
    index: 1
date: 01 July 2026
bibliography: paper.bib
---

# Summary

FerriteShells.jl is a companion library for the Ferrite.jl finite-element ecosystem to simulate and analyze static or dynamic structural mechanics problems involving non-linear shells. It implements a Reissner-Mindlin shell in a residual form which allows to model:

- thick ($t/L>5$) shells (Figure 1),
- thin ($t/L<10$) shells with MITC treatment to prevent shear locking (Figure 2), and
- geometric and material non-linearities through Rodrigues parametrization of the shell director and strain-energy density function-based hyperelastic material laws (Figure 3).

# Statement of need

Shells are are often described as "structures having one dimension much smaller than the others"; this clearly applies to a wide range of engineered structures and their analysis has been the subject of numerous publications. However, their formulations, analysis and physical response can be complex, with wrinkling, buckling and snapping.

FerriteShells.jl aims to bridge the gap between the formulation and analysis of shell structures by providing the users with assembly functions for classical shell formulations which can readily be combined with Ferrite.jl degrees-of-freedom handler, global assembly and boundary conditions to simulate and analyze shell problems.

Two different shell formulations are commonly used in practice, Kirchhoff--Love and Reissner--Mindlin. The former is a thin shell formulation which assumes that the shell's normal remains perpendicular to the midsurface after deformation, while the latter is a thick shell formulation which allows for transverse shear deformation.

FerriteShells.jl implements a Reissner-Mindlin shell which can be used for both thick and thin shells, with the latter being treated with the MITC method to prevent shear locking.


> [!NOTE]
> We note that in the literature, Reissner-Minlin shells are sometimes refered to a geometrically exact Naghdi shells.
> The main difference is that Reissner--Mindlin shells are obtained though a specialization of the three-dimensional elasticity to the shells's midsurface, while Naghdi shells are obained strating from that midsurface and by postulating the existance of a Cosserat-type surface normal.
> The final expression for the membrane, bending and shear tensor are shared between these two formulations.

# State of the field

Shells formualtion often tighlty couple the mathematical formulation and the software implementation.
This leads to 
Specialized element formulation can be used with classical 3D finite element software, see for example [MoFem]() or [CalculiX](https://www.calculix.de); both of which offer support for shells, altough these are continuum (3D) wedge elements.
There is no specialized treatment of shear or membrane locking (appart from reduced integration), but they allow using 3D hyperelastic formulation.

More general shell formulation through weak-form type have been implemented via the FeniCS ecosystem, see [FEniCS Shells](https://fenics-shells.readthedocs.io/en/latest/#).

# Software design

The overall idea of FerriteShells.jl is to adapt Ferrite.jl `CellValue` and the associated `reinit!` to shells to enable seamless integration within the whole Ferrite.jl ecosystem. Since shells rely on specializing the governing equations to the curvilinear system associated with the structure's midsurface, the new `ShellCellValue` holds classical surface metrics, such as the metric tensor, local curvilinear coordinates, etc. This `ShellCellValue` is the used within assembly functions for the energy, residual and consistent tangent that can be used to construct shell problems. This implementation enables simple integration in the Ferrite.jl environment with minimal code change compared to native Ferrite.jl problems and allows user to become rapidly proficient in assembling and solving shell problems.

```julia
scv = ShellCellValue(ip, ip, qr; mitc)
for cell in CellIterator(dh)
    reinit!(scv, cell)
    sd = shelldofs(cell)
    ...
    assemble!(asm, sd, ke, re)
end
```

FerriteShells.jl relies on the Julia programming language multiple dispatch [] to specialize Ferrite.jl functions to `ShellCellValues` and internally assembly function to dispatch to correct behaviour.
This is especially usefull for the MITC treatment of thin shells, which is implemented as a separate type `AbstractMITC` and specialized to the different MITC methods (MITC4, MITC9, etc.) through multiple dispatch. This allows users to easily switch between different MITC methods by simply changing the `AbstractMITC` type used in the `ShellCellValues`.

Additionally, the multiple dispatch allows future users to extend shear treatments to other methods (i.e. Selective Reduced Integration, etc.).

FerriteShells.jl leverage Julia's automatic differentiation capabilities to model strain-energy-based hyperelastic material model through a very simple interface where the (scalar) strain energy function can be specified through the Cartesian Right Cauchy-Green deformation tensor ($C=F^⊤ F$) which is internally specialized to the curvilinear coordinate system of the shell through the incompressibility assumption. This is particularly handy for users modelling hyperelastic shells who do not require to specialize the material model to the shell's coordinate system themselves.

# Research impact statement

This software package is expected to have a significant impact on the field of computational mechanics, particularly in the analysis and simulation of shell structures. By providing a robust and flexible framework for modeling both thick and thin shells, FerriteShells.jl enables researchers and engineers to tackle complex problems that were previously difficult to address. The integration with Ferrite.jl allows for seamless assembly and solution of shell problems, facilitating the exploration of new design concepts and optimization strategies.

# AI usage disclosure

Most of the software developed in this library was done with help of agentic AI (Claude Opus 4.8) under close supervision from the first author. Software architecture, binding with Ferrite.jl, and general implementation route were defined by the first author.

# Acknowledgements

This software was developed as part of the Holland Hybrid Heart project with file number NWA.1518.22.049 of the research program Onderzoek op Routes door Consortia 2022 – NWA-ORC 2022, which is financed by the Dutch Research Council (NWO), the Dutch Ministry of Education, Culture and Science (OCW), and the Hartstichting (Dutch Heart Foundation).
We also thank [collaborators/contributors] for their feedback and contributions.

# References

