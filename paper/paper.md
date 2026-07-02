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
    affiliation: 1
  - name: Viola Bini
    orcid: 0009-0000-2522-827X
    affiliation: 1
  - name: Mathias Peirlinck
    orcid: 0000-0002-4948-5585
    affiliation: 1
    affiliations:
  - name: Department of Biomechanical Engineering, Faculty of Mechanical Engineering, Delft University of Technology, The Netherlands
    index: 1
date: 01 July 2026
bibliography: paper.bib
---

# Summary

FerriteShells.jl is companions for the Ferrite.jl finite-element library to simulate and analyze static and dynamic structural mechanics problem involving non-linear shells. It implements a Reissner-Mindlin shell in a residual form which allows modelling of:
- thick ($t/L>5$) shells (Figure 1),
- thin ($t/L<10$) shells with MITC treatment to prevent shear locking (Figure 2), and
- geometric and material non-linearities through Rodrigues parametrization of the shell director and strain-energy density function-based hyperelastic material laws (Figure 3).

The overall idea of FerriteShells.jl is to adapt Ferrite.jl `CellValue` and the associated `reinit!` to shells to enable seamless integration within the whole Ferrite.jl ecosystem. Since shells rely on specializing the governing equations to the curvilinear system associated with the structure's midsurface, the new `ShellCellValue` holds classical surface metrics, such as the metric tensor, local curvilinear coordinates, etc. This `ShellCellValue` is the used within assembly functions for the energy, residual and consistent tangent that can be used to construct shell problems. This implementation enables simple integration in the Ferrite.jl environment with minimal code change compared to native Ferrite.jl problems and allows user to become rapidly proficient in assembling and solving shell problems.

FerriteShells.jl relies on Julia’s multiple dispatch programming paradigm [@cite] to specialize Ferrite.jl functions to `ShellCellValues` and internally assembly function to dispatch to correct shear treatment. Additionally, the multiple dispatch allows future users to extend shear treatments to other methods (i.e. Selective Reduced Integration, etc.).

FerriteShells.jl leverage Julia's automatic differentiation capabilities to model strain-energy-based hyperelastic material model through a very simple interface where the (scalar) strain energy function can be specified through the Cartesian Right Cauchy-Green deformation tensor ($C=F^⊤ F$) which is internally specialized to the curvilinear coordinate system of the shell through the incompressibility assumption. This is particularly handy for users modelling hyperelastic shells who do not require to specialize the material model to the shell's coordinate system themselves.

# Statement of need

Shells are are often described as "structures having one dimension much smaller than the others"; this clearly applies to a wide range of engineered structures and their analysis has been the subject of numerous publications [@cite]. However, their formulations, analysis and physical response can be complex, with wrinkling, buckling and snapping.

FerriteShells.jl aims to bridge the gap between the formulation and analysis of shell structures by providing the users with assembly functions for classical shell formulations which can readily be combined with Ferrite.jl degrees-of-freedom handler, global assembly and boundary conditions to simulate and analyze shell problems.



# Acknowledgements

This software was developed as part of the Holland Hybrid Heart project with file number NWA.1518.22.049 of the research program Onderzoek op Routes door Consortia 2022 – NWA-ORC 2022, which is financed by the Dutch Research Council (NWO), the Dutch Ministry of Education, Culture and Science (OCW), and the Hartstichting (Dutch Heart Foundation).
We also thank [collaborators/contributors] for their feedback and contributions.

# References