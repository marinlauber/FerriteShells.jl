using FerriteShells, LinearAlgebra
using CairoMakie
import FerriteShells: Vec

# Cook's membrane — RM HyperelasticShell (Neo-Hookean)
#
# Reference (KL, E=1, ν=1/3, t=1, 32×32 Q4): tip y-deflection ≈ 24.84
# For incompressible Neo-Hookean: E_eff = 3μ, so μ = 1/3 matches E=1.

const μ = 1/3      # shear modulus  →  E_eff = 3μ = 1
const t = 1.0
const n = 32       # mesh density

mat = HyperelasticShell(C -> μ/2 * (tr(C) - 3), t)

corners = [Vec{2}((0.,0.)), Vec{2}((48.,44.)), Vec{2}((48.,60.)), Vec{2}((0.,44.))]
grid    = generate_grid(Quadrilateral, (n, n), corners) |> shell_grid
addfacetset!(grid, "clamped",  x -> norm(x[1]) ≈ 0.0)
addfacetset!(grid, "traction", x -> norm(x[1]) ≈ 48.0)

ip  = Lagrange{RefQuadrilateral,1}()
qr  = QuadratureRule{RefQuadrilateral}(2)
fqr = FacetQuadratureRule{RefQuadrilateral}(2)
scv = ShellCellValues(qr, ip, ip)

dh = DofHandler(grid)
add!(dh, :u, ip^3); add!(dh, :θ, ip^2); close!(dh)

K   = allocate_matrix(dh)
f   = zeros(ndofs(dh))
n_el = ndofs_per_cell(dh)
ke  = zeros(n_el, n_el); re = zeros(n_el)
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

@time u_sol = K \ f

# Extract per-node displacements (celldofs field order: 3 u DOFs then 2 θ DOFs per node)
n_nodes  = getnnodes(grid)
ux = zeros(n_nodes); uy = zeros(n_nodes)
counted = falses(n_nodes)
for cell in CellIterator(dh)
    cd = celldofs(cell)
    for (I, nid) in enumerate(getnodes(cell))
        counted[nid] && continue
        counted[nid] = true
        ux[nid] = u_sol[cd[3I-2]]
        uy[nid] = u_sol[cd[3I-1]]
    end
end
umag = sqrt.(ux.^2 .+ uy.^2)

tip_uy = maximum(uy)
@show tip_uy

# Plot reference and deformed mesh
ref_pts    = [Point2f(grid.nodes[i].x[1], grid.nodes[i].x[2]) for i in 1:n_nodes]
deform_pts = [Point2f(grid.nodes[i].x[1] + ux[i], grid.nodes[i].x[2] + uy[i]) for i in 1:n_nodes]
q4_conns   = [cell.nodes for cell in grid.cells]
clim       = (0.0, maximum(umag))

fig = Figure(size=(1100, 500))
for (col, (pts, title)) in enumerate([(ref_pts, "Reference"), (deform_pts, "Deformed (scale=1)")])
    ax = Axis(fig[1,col], title=title, xlabel="x", ylabel="y", aspect=DataAspect())
    for conn in q4_conns
        poly!(ax, Point2f[pts[n] for n in conn],
              color=mean(umag[collect(conn)]),
              colormap=:viridis, colorrange=clim,
              strokecolor=:black, strokewidth=0.3)
    end
end
Colorbar(fig[1,3], colormap=:viridis, limits=clim, label="|u| displacement")

save("dev/cook_hyperelastic_disp.png", fig)
println("Saved dev/cook_hyperelastic_disp.png  (tip u_y = $tip_uy, ref ≈ 24.84)")

# Also write VTK for ParaView inspection
VTKGridFile("dev/cook_hyperelastic_RM", dh) do vtk
    write_solution(vtk, dh, u_sol)
end
println("Saved dev/cook_hyperelastic_RM.vtu")
