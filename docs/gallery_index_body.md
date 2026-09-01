# Code gallery

This page gives an overview of the code gallery. Each entry is a complete, runnable
program that is executed when the documentation is built, so the numbers and figures
shown on the pages are produced by the code above them.

The sources live in [`docs/src/literate-gallery`](https://github.com/marinlauber/FerriteShells.jl/tree/master/docs/src/literate-gallery)
and are rendered with [Literate.jl](https://github.com/fredrikekre/Literate.jl). Every
page also carries a comment-free "Plain program" version at the bottom that can be copied
and run as-is.

!!! note "Contribute to the gallery!"
    If you use FerriteShells for something — a published result, an unusual boundary
    condition, a solver trick — please add it here. See
    [Adding a gallery item](@ref) below.

---

#### [Linear elastic shell](linear-elasticity.md)

Cook's membrane solved as a Reissner–Mindlin shell embedded in 3D: a `ShellCellValues`
assembly loop over a Q9 mesh, an edge traction, and membrane/bending/shear strains
exported to VTK through an `L2Projector`.

---

## Adding a gallery item

A gallery item is a single Literate.jl script; everything else is bookkeeping. Say the new
item is called `pinched-cylinder`:

**1. Write the script** as `docs/src/literate-gallery/pinched-cylinder.jl`. Ordinary Julia
code, with comment lines carrying the prose:

```julia
# # [Pinched cylinder](@id gallery-pinched-cylinder)
#
# ## Introduction
#
# A short description of the problem and what the example demonstrates.

using FerriteShells

# Comments between code blocks become the text between them.
grid = get_ferrite_grid("cylinder.inp")
```

Literate directives worth knowing: `#md` lines appear only in the markdown output, `#hide`
lines run but are not shown, and `#src` lines are dropped from the output entirely (useful
for `@test` assertions that guard the example against regressions).

Give the page an `@id` of the form `gallery-<name>` so that other pages can link to it with
`[Pinched cylinder](@ref gallery-pinched-cylinder)`.

**2. End the script with the plain-program footer.** `docs/generate.jl` runs
`Literate.script` and substitutes the comment-free source for `@__CODE__`:

```julia
#md # ## [Plain program](@id gallery-pinched-cylinder-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`pinched-cylinder.jl`](pinched-cylinder.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
```

**3. Register the page** in the `pages` list of `docs/make.jl`:

```julia
"Code gallery" => [
    "Code gallery overview" => "gallery/index.md",
    "gallery/linear-elasticity.md",
    "gallery/pinched-cylinder.md",
],
```

**4. Describe it on this page** by appending a `---`-separated section to
`docs/gallery_index_body.md` (the file you are reading; `docs/generate.jl` copies it to
`src/gallery/index.md`, which is generated and not tracked by git):

```markdown
---

#### [Pinched cylinder](pinched-cylinder.md)

One or two sentences on what the example does and which features it exercises.
```

**5. Add any new dependencies** to `docs/Project.toml`. The scripts are executed while the
documentation is built, so a package that is not in that environment will fail the build,
and an unregistered or local package cannot be resolved in CI at all. If an example
genuinely cannot run — it needs a mesh that is not in the repository, or hours of
computation — add its file name to `DONT_EXECUTE` in `docs/generate.jl`; its code is then
rendered but never run.

### Figures

Images placed next to the script in `docs/src/literate-gallery/` are copied to the output
directory, so they can be referenced by bare file name:

```julia
# ![](pinched-cylinder.png)
```

Note that `.gitignore` excludes `*.png` repository-wide, so a new figure has to be added
with `git add -f`. Images that already live in `docs/src/images/` are referenced relatively
instead, as `![](../images/pinched_cylinder.png)`.

### Building locally

```
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The rendered pages end up in `docs/build/`. Any VTK files an example writes are deleted
again by `docs/generate.jl` and are not deployed.
