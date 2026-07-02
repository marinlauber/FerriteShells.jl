using FerriteShells, LinearAlgebra
using CairoMakie
import FerriteShells: Vec

function cook_nh_solve(n_mesh)
    μ = 1.0; t = 1.0
    mat = HyperelasticShell(C -> μ/2 * (tr(C) - 3), t)

    corners = [Vec{2}((0.,0.)), Vec{2}((48.,44.)), Vec{2}((48.,60.)), Vec{2}((0.,44.))]
    grid    = generate_grid(Quadrilateral, (n_mesh, n_mesh), corners) |> shell_grid
    addfacetset!(grid, "clamped",  x -> norm(x[1]) ≈ 0.0)
    addfacetset!(grid, "traction", x -> norm(x[1]) ≈ 48.0)

    ip  = Lagrange{RefQuadrilateral,1}()
    qr  = QuadratureRule{RefQuadrilateral}(2)
    fqr = FacetQuadratureRule{RefQuadrilateral}(2)
    scv = ShellCellValues(qr, ip, ip)

    dh = DofHandler(grid)
    add!(dh, :u, ip^3); add!(dh, :θ, ip^2); close!(dh)

    K = allocate_matrix(dh)
    f = zeros(ndofs(dh))

    n_el = ndofs_per_cell(dh)
    ke = zeros(n_el, n_el); re = zeros(n_el)
    asm = start_assemble(K, f)
    for cell in CellIterator(dh)
        fill!(ke, 0); fill!(re, 0)
        reinit!(scv, cell)
        sd  = shelldofs(cell)
        u_e = zeros(n_el)
        membrane_residuals_RM!(re, scv, u_e, mat)
        bending_residuals_RM!(re, scv, u_e, mat)
        membrane_tangent_RM!(ke, scv, u_e, mat)
        bending_tangent_RM!(ke, scv, u_e, mat)
        assemble!(asm, sd, ke, re)
    end

    assemble_traction!(f, dh, getfacetset(grid,"traction"), ip, fqr, Vec{3}((0., 1/16, 0.)))

    dbc = ConstraintHandler(dh)
    add!(dbc, Dirichlet(:u, getfacetset(grid,"clamped"), x -> zeros(3), [1,2,3]))
    add!(dbc, Dirichlet(:θ, getfacetset(grid,"clamped"), x -> zeros(2), [1,2]))
    close!(dbc); apply!(K, f, dbc)

    u_sol = K \ f
    return grid, dh, u_sol
end

grid, dh, u_sol = cook_nh_solve(32)

# Extract per-node u_x, u_y from the :u field (DOFs 1,2,3 per node)
n_nodes  = getnnodes(grid)
ux_nodes = zeros(n_nodes)
uy_nodes = zeros(n_nodes)
counted  = falses(n_nodes)

for cell in CellIterator(dh)
    node_ids = getnodes(cell)
    cd = celldofs(cell)   # field order: 3*n_base u DOFs then 2*n_base θ DOFs
    n_base = length(node_ids)
    for (I, nid) in enumerate(node_ids)
        counted[nid] && continue
        counted[nid] = true
        ux_nodes[nid] = u_sol[cd[3I-2]]   # u_x for node I in :u field
        uy_nodes[nid] = u_sol[cd[3I-1]]   # u_y for node I in :u field
    end
end

# Reference and deformed node positions
scale = 1.0
ref_xy   = [Point2f(grid.nodes[i].x[1], grid.nodes[i].x[2]) for i in 1:n_nodes]
deform   = [Point2f(grid.nodes[i].x[1] + scale*ux_nodes[i],
                    grid.nodes[i].x[2] + scale*uy_nodes[i]) for i in 1:n_nodes]
umag     = sqrt.(ux_nodes.^2 .+ uy_nodes.^2)

# Connectivity for Q4 cells
q4_conns = [cell.nodes for cell in grid.cells]

fig = Figure(size=(1100, 500))
clim = (0.0, maximum(umag))

for (col, (pts, title)) in enumerate([(ref_xy, "Reference"), (deform, "Deformed (scale=1)")])
    ax = Axis(fig[1,col], title=title, xlabel="x", ylabel="y", aspect=DataAspect())
    for conn in q4_conns
        xs = Float32[pts[n][1] for n in conn]
        ys = Float32[pts[n][2] for n in conn]
        cs = Float32[umag[n]   for n in conn]
        poly!(ax, Point2f.(xs, ys), color=sum(cs)/4,
              colormap=:viridis, colorrange=clim, strokecolor=:black, strokewidth=0.4)
    end
end

Colorbar(fig[1,3], colormap=:viridis, limits=clim, label="|u| displacement")
save("dev/cook_hyperelastic_disp.png", fig)
println("Saved to dev/cook_hyperelastic_disp.png")
println("Max |u| = ", maximum(umag))
println("Tip u_y ≈ ", maximum(uy_nodes))
