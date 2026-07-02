using FerriteShells, LinearAlgebra, Printf, WriteVTK
using FerriteGmsh   # re-exports `gmsh` and `togrid`

# Kirigami parachute mesh — flat circular shell plate with concentric arc cuts.
#
# Inspired by deployable concentric-cut kirigami (Nature 2025, s41586-025-09515-9):
# a disk patterned with rings of arc-shaped cuts separated by uncut `ligaments`.
# Adjacent rings are angularly staggered by half a cut pitch so the ligaments
# never line up radially. Under a central (payload) pull the strips between the
# cuts rotate out of plane and the disk deploys into a 3D canopy.
#
# The cuts are true zero-width slits: each arc is embedded into the disk via
# `occ.fragment`, then gmsh's `Crack` plugin duplicates the coincident nodes along
# the arc so the two sides separate mechanically. The result is an all-Q9 Ferrite
# grid embedded in 3D via `shell_grid`.

"""
    make_kirigami_parachute_grid(; R, n_rings, n_cuts_per_ring, ligament_angle,
                                   r_inner, r_outer, h, stagger)

Build a flat circular Q9 shell grid of radius `R` with `n_rings` concentric rings
of zero-width arc cuts, `n_cuts_per_ring` cuts per ring. Each cut spans the angular
pitch `2π/n_cuts_per_ring` minus a fixed uncut gap `ligament_angle` (radians).
Rings are spaced uniformly between radii `r_inner` and `r_outer`. `h` is the target
element size. Odd rings are staggered by half a pitch when `stagger=true`.

Tags the outer rim facets as `"edge"` and the centre node as `"center"`.
"""
function make_kirigami_parachute_grid(; R=1.0, n_rings=4, n_cuts_per_ring=4,
                                      ligament_angle=0.15,
                                      r_inner=0.20R, r_outer=0.92R,
                                      h=0.03R, stagger=true)
    pitch = 2π / n_cuts_per_ring
    @assert ligament_angle < pitch "ligament_angle ($ligament_angle) must be < pitch ($pitch)"

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)   # full Q9, not Q8
    gmsh.option.setNumber("Mesh.Algorithm", 8)               # frontal-Delaunay for quads
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)  # blossom → all-quad
    gmsh.option.setNumber("Mesh.MeshSizeMin", h)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.model.add("kirigami_parachute")

    occ = gmsh.model.occ
    disk = occ.addDisk(0.0, 0.0, 0.0, R, R)
    c    = occ.addPoint(0.0, 0.0, 0.0)   # arc centre, shared by all cuts

    # One zero-width arc cut at radius `r`, angular span [a1, a2].
    function arc_curve(r, a1, a2)
        p1 = occ.addPoint(r*cos(a1), r*sin(a1), 0.0)
        p2 = occ.addPoint(r*cos(a2), r*sin(a2), 0.0)
        occ.addCircleArc(p1, c, p2)
    end

    radii = range(r_inner, r_outer; length=n_rings)
    Δarc  = pitch - ligament_angle
    arcs  = Int[]
    for (k, r) in enumerate(radii)
        θ0 = (stagger && isodd(k)) ? pitch/2 : 0.0
        for j in 0:n_cuts_per_ring-1
            a1 = θ0 + j*pitch + ligament_angle/2
            push!(arcs, arc_curve(r, a1, a1 + Δarc))
        end
    end

    # Embed the arcs into the disk; track the resulting cut curves through the
    # fragment so they can be tagged and handed to the Crack plugin.
    _, outmap = occ.fragment([(2, disk)], [(1, a) for a in arcs])
    occ.synchronize()
    cut_curves = unique(Int[t for m in outmap[2:end] for (d, t) in m if d == 1])

    surfs  = [t for (d, t) in gmsh.model.getEntities(2)]
    gmsh.model.addPhysicalGroup(2, surfs, -1, "canopy")
    cut_pg = gmsh.model.addPhysicalGroup(1, cut_curves, -1, "cuts")
    gmsh.model.mesh.embed(0, [c], 2, surfs[1])   # force a mesh node at the centre
    for s in surfs; gmsh.model.mesh.setRecombine(2, s); end
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)

    # Split the coincident nodes along the cuts into a real zero-width slit.
    gmsh.plugin.setNumber("Crack", "Dimension", 1)
    gmsh.plugin.setNumber("Crack", "PhysicalGroup", cut_pg)
    gmsh.plugin.run("Crack")

    # Verify all-quad before handing to Ferrite (mixed grids break the Q9 shell).
    etypes, _ = gmsh.model.mesh.getElements(2)
    @assert all(t -> t in (10, 16), etypes) "non-quad elements present: $etypes (10=Q9, 16=Q8)"

    grid2d = togrid()
    gmsh.finalize()

    grid = shell_grid(grid2d)
    addfacetset!(grid, "edge",   x -> isapprox(norm(x[1:2]), R; atol=1.5h))
    # The uncut central hub (inside the first ring of cuts) is the payload
    # attachment patch; clamping it fixes the parachute against all rigid motion.
    addnodeset!(grid,  "center", x -> norm(x[1:2]) < 0.6r_inner)
    return grid
end

function assemble_internal!(K, g, u, dh, scv, mat)
    n_e = ndofs_per_cell(dh)
    ke  = zeros(n_e, n_e)
    re  = zeros(n_e)
    asm = start_assemble(K, g)
    for cell in CellIterator(dh)
        fill!(ke, 0.0); fill!(re, 0.0)
        reinit!(scv, cell)
        u_e = u[shelldofs(cell)]
        membrane_tangent_RM!(ke, scv, u_e, mat)
        bending_tangent_RM!(ke, scv, u_e, mat)
        membrane_residuals_RM!(re, scv, u_e, mat)
        bending_residuals_RM!(re, scv, u_e, mat)
        assemble!(asm, shelldofs(cell), ke, re)
    end
end

internal_energy(u, dh, scv, mat) = sum(CellIterator(dh)) do cell
    reinit!(scv, cell)
    FerriteShells.energy_RM(u[shelldofs(cell)], scv, mat)
end

R    = 1.0
grid = make_kirigami_parachute_grid(; R=R)
@printf("kirigami parachute: %d Q9 cells, %d nodes\n", getncells(grid), getnnodes(grid))

ip  = Lagrange{RefQuadrilateral, 2}()
qr  = QuadratureRule{RefQuadrilateral}(3)
scv = ShellCellValues(qr, ip, ip; mitc=MITC9)
mat = LinearElastic(1.0e6, 0.3, 2.0e-3)

dh = DofHandler(grid); add!(dh, :u, ip^3); add!(dh, :θ, ip^2); close!(dh)
n_base = getnbasefunctions(ip)

# Uniform transverse (+z) distributed load — the drag pushing on the canopy.
# Dead load (direction fixed in space), so the force vector is assembled once.
f_ext = zeros(ndofs(dh))
fe = zeros(5n_base)
q  = Vec{3}((0.0, 0.0, 5.0e-5))
for cell in CellIterator(dh)
    fill!(fe, 0.0)
    reinit!(scv, cell)
    for qp in 1:getnquadpoints(scv)
        ξ  = scv.qr.points[qp]; dΩ = scv.detJdV[qp]
        for I in 1:n_base
            NI = Ferrite.reference_shape_value(ip, ξ, I)
            @views fe[5I-4:5I-2] .+= NI * q * dΩ
        end
    end
    @views f_ext[shelldofs(cell)] .+= fe
end

# Fix the central hub against any motion (translations + director rotations).
dbc = ConstraintHandler(dh)
add!(dbc, Dirichlet(:u, getnodeset(grid, "center"), x -> zeros(3), [1, 2, 3]))
add!(dbc, Dirichlet(:θ, getnodeset(grid, "center"), x -> zeros(2), [1, 2]))
close!(dbc); Ferrite.update!(dbc, 0.0)

# Load-controlled Newton-Raphson with load steps and an energy line search.
# The flat petals carry the transverse load almost entirely in bending at first
# and stiffen geometrically as they deflect, so plain Newton overshoots from the
# flat reference (residual-norm Armijo is unreliable here). The merit function is
# the total potential Π = E_int − λ·F·u; the Newton direction is a descent
# direction for Π when K is PD, so backtracking on Π damps the overshoot.
K       = allocate_matrix(dh)
g       = zeros(ndofs(dh))
u       = zeros(ndofs(dh))
Δu      = zeros(ndofs(dh))
u_trial = zeros(ndofs(dh))
free    = dbc.free_dofs

pvd = paraview_collection("kirigami_parachute")
VTKGridFile("kirigami_parachute-0", dh) do vtk
    write_solution(vtk, dh, u); pvd[0.0] = vtk
end

nsteps = 6
let
@time for (i, λ) in enumerate(range(1/nsteps, 1.0; length=nsteps))
    fnorm = max(norm(@views (λ .* f_ext)[free]), eps())
    newton_itr = 0; rnorm = Inf
    while true
        newton_itr += 1
        assemble_internal!(K, g, u, dh, scv, mat)
        g .-= λ .* f_ext
        apply_zero!(K, g, dbc)
        rnorm = norm(@views g[free])
        rnorm < 1e-6 * fnorm && break
        newton_itr > 40 && (@warn "step $i did not converge (‖r‖=$rnorm)"; break)
        Δu .= K \ g
        apply_zero!(Δu, dbc)
        # Energy line search: accept the largest α with Π decreasing (Armijo).
        Π0    = internal_energy(u, dh, scv, mat) - λ * dot(f_ext, u)
        slope = -dot(g, Δu)            # = ∇Π·(−Δu), < 0 since K is PD
        α = 1.0
        for _ in 1:25
            @. u_trial = u - α * Δu
            Πα = internal_energy(u_trial, dh, scv, mat) - λ * dot(f_ext, u_trial)
            Πα ≤ Π0 + 1e-4 * α * slope && break
            α *= 0.5
        end
        @. u = u - α * Δu
    end
    @printf("load step %d/%d (λ=%.3f): %2d Newton iters, ‖r‖/‖f‖=%.2e\n",
            i, nsteps, λ, newton_itr, rnorm / fnorm)
    d, G3 = director_field(dh, scv, u)
    VTKGridFile("kirigami_parachute-$i", dh) do vtk
        write_solution(vtk, dh, u)
        Ferrite.write_node_data(vtk, d,  "director")
        Ferrite.write_node_data(vtk, G3, "G3")
        pvd[Float64(i)] = vtk
    end
end; end
close(pvd)

# :u is the first (3-component) field, grouped by node in celldofs → z is cd[3I].
uz_max = maximum(CellIterator(dh)) do cell
    cd = celldofs(cell)
    maximum(abs(u[cd[3I]]) for I in 1:n_base)
end
@printf("max |u_z| = %.4e at q_z = %.2e (nonlinear)\n", uz_max, q[3])
println("wrote kirigami_parachute.pvd")
