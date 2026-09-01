# # [Linear elastic shell](@id gallery-linear-elastic-shell)
#
# ![](../images/cooks_membrane.png)
#
# *Figure 1*: Cook's membrane, solved as a shell embedded in 3D and coloured by the
# vertical displacement.
#
#
# ## Introduction
#
# Cook's membrane is a classic bending-dominated plane-stress benchmark: a tapered,
# clamped panel loaded by a shear traction on its free edge. Here we solve it with a
# Reissner–Mindlin shell embedded in three dimensions, which is the flat-shell limit of
# the general formulation: the mesh is planar, so the membrane and bending responses
# decouple and only the membrane part carries the load.
#
# The point of the example is the assembly pattern rather than the physics — it shows the
# minimal set of pieces (`ShellCellValues`, `shelldofs`, the `_RM!` residual/tangent
# functions) that every FerriteShells program is built from.

# ## Commented program
#
#md # You can also find the same program without comments at the end of the page,
#md # see [Plain program](@ref gallery-linear-elastic-shell-plain-program).
using FerriteShells
using WriteVTK

# We first have to define a mesh. `generate_grid` gives us a two-dimensional grid, which
# `shell_grid` embeds into 3D by adding a zero third coordinate to every node.

function create_cook_grid(nx, ny; primitive = Quadrilateral)
    corners = [
        Vec{2}((0.0, 0.0)), Vec{2}((48.0, 44.0)),
        Vec{2}((48.0, 60.0)), Vec{2}((0.0, 44.0)),
    ]
    return generate_grid(primitive, (nx, ny), corners) |> shell_grid # embed it into a 3D space
end

# The assembly loop is the standard FerriteShells pattern: `reinit!` the shell values on
# the cell, extract the element displacements with [`shelldofs`](@ref) (*not* `celldofs`,
# see below), and add the membrane and bending contributions to the element tangent and
# residual.

function assemble_membrane!(K, r, dh, scv, u, mat)
    n = ndofs_per_cell(dh)
    ke = zeros(n, n)
    re = zeros(n)
    assembler = start_assemble(K, r)
    for cell in CellIterator(dh)
        fill!(ke, 0.0); fill!(re, 0.0)
        reinit!(scv, cell) # prepares reference geometry
        u_e = u[shelldofs(cell)]
        membrane_tangent_RM!(ke, scv, u_e, mat)
        bending_tangent_RM!(ke, scv, u_e, mat)
        membrane_residuals_RM!(re, scv, u_e, mat)
        bending_residuals_RM!(re, scv, u_e, mat)
        assemble!(assembler, shelldofs(cell), ke, re)
    end
    return K, r
end

# A helper to compute the membrane, bending, and shear strains at the quadrature points
# for postprocessing. `shell_strains` returns the in-plane (2×2) tensors, which
# [`embed23`](@ref) lifts into 3D so that they can be written to VTK.

function compute_strains(dh, scv, u)
    n_qp = getnquadpoints(scv)
    n_cells = getncells(dh.grid)
    E_mem = [Vector{SymmetricTensor{2, 3, Float64, 6}}(undef, n_qp) for _ in 1:n_cells]
    kappa = [Vector{SymmetricTensor{2, 3, Float64, 6}}(undef, n_qp) for _ in 1:n_cells]
    gamma = [Vector{Vec{3, Float64}}(undef, n_qp) for _ in 1:n_cells]
    for cell in CellIterator(dh)
        reinit!(scv, cell)
        u_e = u[shelldofs(cell)]
        id = cellid(cell)
        @inbounds for qp in 1:n_qp
            E, κ, γ = shell_strains(scv, qp, u_e)
            E_mem[id][qp] = embed23(E)
            kappa[id][qp] = embed23(κ)
            gamma[id][qp] = Vec{3}((γ[1], γ[2], 0.0))
        end
    end
    return E_mem, kappa, gamma
end

# Now for the program itself. We start with the mesh; `QuadraticQuadrilateral` gives the
# nine-noded cells that the quadratic interpolation below needs.

grid = create_cook_grid(32, 16; primitive = QuadraticQuadrilateral)

# Facetsets for the boundary conditions: the panel is clamped at `x = 0` and loaded on the
# `x = 48` edge.

addfacetset!(grid, "clamped", x -> norm(x[1]) ≈ 0.0)
addfacetset!(grid, "traction", x -> norm(x[1]) ≈ 48.0)

# The interpolation and the quadrature rule. Q9 (`Lagrange{RefQuadrilateral, 2}`) is used
# here because it resolves the full curvature tensor; Q4 only captures the twist.

ip = Lagrange{RefQuadrilateral, 2}() # Q9
qr = QuadratureRule{RefQuadrilateral}(3)
fqr = FacetQuadratureRule{RefQuadrilateral}(3)

# [`ShellCellValues`](@ref) replaces Ferrite's `CellValues` and carries the covariant basis
# vectors and the element director frame that the shell kinematics need.

scv = ShellCellValues(qr, ip, ip)

# Degrees of freedom, five per node: three displacements `:u` and two rotations `:θ`.
#
# !!! note "shelldofs vs celldofs"
#     The two-field `DofHandler` does not order the DOFs as the interleaved
#     `[u₁, u₂, u₃, φ₁, φ₂, …]` layout that the assembly functions expect. Always extract
#     the element DOFs with [`shelldofs`](@ref), never with `celldofs`.

dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

# The linear elastic material, given by Young's modulus, Poisson's ratio, and the shell
# thickness.

mat = LinearElastic(1.0, 1 / 3, 1.0)

# Clamped edge: all three displacements and both rotations are fixed.

dbc = ConstraintHandler(dh)
add!(dbc, Dirichlet(:u, getfacetset(grid, "clamped"), x -> zero(x), [1, 2, 3]))
add!(dbc, Dirichlet(:θ, getfacetset(grid, "clamped"), x -> [0.0, 0.0], [1, 2]))
close!(dbc)

# Stiffness matrix and residual vector construction and assembly. Because we linearise
# about the undeformed state the residual vanishes and `Ke` is the linear stiffness.

Ke = allocate_matrix(dh)
f = zeros(ndofs(dh))
assemble_membrane!(Ke, f, dh, scv, zeros(ndofs(dh)), mat)

# Traction force assembly: a total force of 1 N on the free edge, spread over its length
# of 16.

assemble_traction!(f, dh, getfacetset(grid, "traction"), ip, fqr, (0.0, 1 / 16, 0.0))

# Apply the boundary conditions and solve; `\` picks a suitable linear solver.

apply!(Ke, f, dbc)
ue = Ke \ f

# Extract the solution at the classic evaluation point, the midpoint of the loaded edge.
# The reference value for the vertical tip displacement is about 23.9.

ph = PointEvalHandler(grid, [Vec{3}((48.0, 52.0, 0.0))])
u_eval = first(evaluate_at_points(ph, dh, ue, :u))

# Finally, write the displacements and the projected strains to VTK. The strains live at
# the quadrature points, so they are pushed to the nodes with an `L2Projector` first.

proj = L2Projector(ip, grid)

VTKGridFile("cooks_membrane", dh) do vtk
    write_solution(vtk, dh, ue)
    E_mem, κ, γ = compute_strains(dh, scv, ue)
    write_projection(vtk, proj, project(proj, E_mem, qr), "E_membrane")
    write_projection(vtk, proj, project(proj, κ, qr), "kappa_bending")
    write_projection(vtk, proj, project(proj, γ, qr), "gamma_shear")
end

#md # ## [Plain program](@id gallery-linear-elastic-shell-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`linear-elasticity.jl`](linear-elasticity.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
