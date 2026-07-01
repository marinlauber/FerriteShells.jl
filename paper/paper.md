---
title: 'FerriteShells.jl: A Julia Package for Nonlinear Shell Finite Elements Built on Ferrite.jl'
tags:
  - Julia
  - finite elements
  - shell elements
  - fluid-structure interaction
  - computational mechanics
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
  - name: Faculty of [Department], Delft University of Technology, The Netherlands
    index: 1
date: 01 July 2026
bibliography: paper.bib
---

# Summary

A few sentences describing the high-level functionality and purpose of the
software for a diverse, non-specialist audience. What problem does the
software solve, and who is it for? (E.g.: FerriteShells.jl extends the
Ferrite.jl finite element framework with nonlinear geometrically exact shell
and membrane elements, enabling large-deformation structural analysis in
Julia, with particular applicability to fluid-structure interaction problems
such as membrane wings and sails.)

# Statement of need

A clear description of what problem the software is solving, why existing
tools are insufficient, and who the target audience is (researchers,
students, industry). Reference related/competing software packages here
(e.g. other FE packages, commercial shell solvers) and explain how this
package differs or fills a gap. Cite prior work and any papers that have
already used the software.

# Mathematical background (optional)

A brief description of the underlying formulation if relevant — e.g. the
shell kinematics, element formulation, or solution scheme implemented in
the package — enough to orient a reader but not a full derivation.

# Functionality

Describe the main features: what element types are implemented, what
solvers/nonlinear schemes are supported, how it interfaces with Ferrite.jl,
any coupling capabilities (e.g. with immersed-boundary CFD codes), and
example use cases or benchmarks that validate the implementation.

# Example usage (optional)

```julia
using Ferrite, FerriteShells

# minimal example demonstrating typical usage
```

# Acknowledgements

This software was developed as part of the project Holland Hybrid Heart with file number NWA.1518.22.049 of the research program Onderzoek op Routes door Consortia 2022 – NWA-ORC 2022, which is financed by the Dutch Research Council (NWO), the Dutch Ministry of Education, Culture and Science (OCW), and the Hartstichting (Dutch Heart Foundation).
We also thank [collaborators/contributors] for their feedback and
contributions.

# References