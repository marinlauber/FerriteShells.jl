using FerriteShells, FerriteGmsh, LinearAlgebra, Printf, WriteVTK
const gmsh = FerriteGmsh.gmsh

# Slit annular plate (Sze, Liu & Lo 2004) 10.1016/j.finel.2003.11.001
function slit_annular_grid(a, b, n_r, n_q)
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    g = gmsh.model.geo
    c   = g.addPoint(0, 0, 0)
    ang = (0.0, π/2, π, 3π/2, 2π)            # 2π and 0 coincide → the slit
    ip  = [g.addPoint(a*cos(θ), a*sin(θ), 0) for θ in ang]
    op  = [g.addPoint(b*cos(θ), b*sin(θ), 0) for θ in ang]
    radials = [g.addLine(ip[k], op[k]) for k in 1:5]
    surfs   = Int[]
    for k in 1:4
        ia = g.addCircleArc(ip[k], c, ip[k+1])
        oa = g.addCircleArc(op[k], c, op[k+1])
        s  = g.addPlaneSurface([g.addCurveLoop([radials[k], oa, -radials[k+1], -ia])])
        push!(surfs, s)
        g.mesh.setTransfiniteCurve(ia, n_q+1); g.mesh.setTransfiniteCurve(oa, n_q+1)
        g.mesh.setTransfiniteCurve(radials[k], n_r+1); g.mesh.setTransfiniteCurve(radials[k+1], n_r+1)
        g.mesh.setTransfiniteSurface(s); g.mesh.setRecombine(2, s)
    end
    g.synchronize()
    gmsh.model.addPhysicalGroup(2, surfs, -1, "plate")
    gmsh.model.addPhysicalGroup(1, [radials[1]], -1, "clamped")   # θ = 0 edge
    gmsh.model.addPhysicalGroup(1, [radials[5]], -1, "loaded")    # θ = 2π edge
    gmsh.model.mesh.generate(2); gmsh.model.mesh.setOrder(2)
    grid = togrid(); gmsh.finalize()
    return shell_grid(grid)                    # embed the planar Q9 grid into 3D (z = 0)
end

# Residual + tangent via ForwardDiff on energy_RM, required because theexplciit MITC variant is assymetric
function assemble_global!(K, r, dh, scv, u, mat)
    n_e = ndofs_per_cell(dh); ke = zeros(n_e, n_e); re = zeros(n_e)
    asm = start_assemble(K, r)
    for cell in CellIterator(dh)
        fill!(ke, 0.0); fill!(re, 0.0)
        reinit!(scv, cell)
        u_e = u[shelldofs(cell)]
        residuals_RM_FD!(re, scv, u_e, mat)
        tangent_RM_FD!(ke, scv, u_e, mat)
        assemble!(asm, shelldofs(cell), ke, re)
    end
end

# Global u_z DOF of a node, from the interleaved :u block (3 dofs/node, node-major).
function nodal_uz_dof(dh, node_id)
    for cell in CellIterator(dh), (I, gid) in enumerate(getnodes(cell))
        gid == node_id && return celldofs(cell)[3I]
    end
    error("node $node_id not found in any cell")
end
nodal_uz(dh, u, node_id) = u[nodal_uz_dof(dh, node_id)]

# geometry, material and reference load (Sze, Liu & Lo 2004): a=6, b=10, t=0.03,
# E=21e6, ν=0, max transverse line load q_max=0.8 (force / length) at the free edge.
a, b, t = 6.0, 10.0, 0.03
mat     = LinearElastic(21.0e6, 0.0, t)
q_max   = 0.8

# mesh and interpolation
grid = slit_annular_grid(a, b, 4, 10)
ip   = Lagrange{RefQuadrilateral, 2}()
qr   = QuadratureRule{RefQuadrilateral}(3)
fqr  = FacetQuadratureRule{RefQuadrilateral}(3)
scv  = ShellCellValues(qr, ip, ip; mitc=MITC9)

# degrees of freedom
dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

# boundary conditions: fully clamp the θ = 0 slit edge
ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "clamped"), x -> zeros(3), [1,2,3]))
add!(ch, Dirichlet(:θ, getfacetset(grid, "clamped"), x -> zeros(2), [1,2]))
close!(ch)

# corner evaluation nodes on the loaded edge: A = outer (r=b), B = inner (r=a)
loaded_nodes = unique(vcat([collect(Ferrite.facets(grid.cells[c])[f])
                            for (c, f) in getfacetset(grid, "loaded")]...))
radius(nid)  = norm(grid.nodes[nid].x)
node_A = loaded_nodes[argmax(radius.(loaded_nodes))]
node_B = loaded_nodes[argmin(radius.(loaded_nodes))]

# reference load vector (full line load); the actual load is λ·F_ext
N_dofs = ndofs(dh)
f_ext  = zeros(N_dofs)
assemble_traction!(f_ext, dh, getfacetset(grid, "loaded"), ip, fqr, Vec{3}((0.0, 0.0, q_max)))

# Load-controlled Newton–Raphson with automatic load-increment cutback (Sze et al. 2004)
w_dof    = nodal_uz_dof(dh, node_A)   # outer free corner, for monitoring only
tol      = 1e-6
max_iter = 16
Δλ       = 0.05
Δλ_min   = 1e-6

K = allocate_matrix(dh)
r = zeros(N_dofs)
Δ = zeros(N_dofs)

pvd = paraview_collection("slit_annular_plate")
println("Slit annular plate (load control + cutback, $(getncells(grid)) cells)")
println("     λ    |   Δλ     |  u_z(A,outer) | u_z(B,inner) | iters")
VTKGridFile("slit_annular_plate-0", dh) do vtk
    write_solution(vtk, dh, zeros(N_dofs)); pvd[0.0] = vtk
end

u = zeros(N_dofs); u_conv = zeros(N_dofs); λ = 0.0; n_out = 0
trace = Tuple{Float64,Float64,Float64}[(0.0, 0.0, 0.0)]
while λ < 1.0 - 1e-12
    global u, u_conv, λ, Δλ, n_out
    Δλ    = min(Δλ, 1.0 - λ)
    λ_try = λ + Δλ
    u .= u_conv                              # restart this attempt from last converged state
    converged = false; n_iter = 0
    for iter in 1:max_iter
        assemble_global!(K, r, dh, scv, u, mat)
        @. r = r - λ_try * f_ext             # R(u) = R_int(u) − λ·f_ext
        apply_zero!(K, r, ch)
        res = norm(r)
        isfinite(res) && res < tol && (converged = true; n_iter = iter - 1; break)
        n_iter = iter
        Δ .= K \ r; apply_zero!(Δ, ch)
        u .-= Δ
    end
    if !converged
        Δλ /= 4
        Δλ < Δλ_min && (@warn "load increment below Δλ_min at λ=$λ; aborting"; break)
        continue
    end
    λ = λ_try; u_conv .= u; n_out += 1
    w_A, w_B = nodal_uz(dh, u, node_A), nodal_uz(dh, u, node_B)
    push!(trace, (λ, w_A, w_B))
    @printf("  %.5f | %.6f | %13.4f | %12.4f | %d\n", λ, Δλ, w_A, w_B, n_iter)
    VTKGridFile("slit_annular_plate-$n_out", dh) do vtk
        write_solution(vtk, dh, u); pvd[float(n_out)] = vtk
    end
    n_iter ≤ 5 && (Δλ *= 1.5) # grow the increment after an easy step
end
vtk_save(pvd)

# Reference (Sze et al. 2004) at full load (λ=1): u_z(A,outer) ≈ 17.5, u_z(B,inner) ≈ 13.9
using Plots
λs, wA, wB = getindex.(trace, 1), getindex.(trace, 2), getindex.(trace, 3)
plot([wA, wB], λs, marker=:o, label=["u_z(A) outer" "u_z(B) inner"],
     xlabel="tip deflection", ylabel="load factor λ", legend=:bottomright)
