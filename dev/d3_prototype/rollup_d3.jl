include(joinpath(@__DIR__, "rm_d3.jl")); using Printf
const L, W, t, nsteps = 10.0, 1.0, 0.1, parse(Int, get(ARGS, 1, "40"))
ip, qr = Lagrange{RefQuadrilateral,2}(), QuadratureRule{RefQuadrilateral}(3)
g2 = generate_grid(QuadraticQuadrilateral, (20, 2), Vec{2}((0.0, 0.0)), Vec{2}((L, W)))
grid = Grid(getcells(g2), [Node(Vec{3}((x[1], x[2], 0.0))) for x in get_node_coordinate.(getnodes(g2))]; facetsets = g2.facetsets)
addnodeset!(grid, "tip", x -> x[1] ≈ L && x[2] ≈ W/2)
s = RMShell(ip, ip, qr; E = 1.2e6, ν = 0.0, t)
dh = DofHandler(grid); add!(dh, :u, ip^3); add!(dh, :ω, ip^3); close!(dh)
Dn = nodal_normals(grid, ip)
ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "left"), x -> zero(x)))
add!(ch, Dirichlet(:ω, getfacetset(grid, "left"), x -> zero(x)))
drilling_constraints!(ch, dh, Dn); close!(ch)
# end moment m = M/W about y, as a load on the ω_y dofs (work conjugate for rotation about a fixed axis)
M_ref = 2π * 1.2e6 * W * t^3 / 12 / L
# FacetValues does not support embedded surfaces (no weighted_normal for 3×2 jacobians): integrate by hand
fqr = FacetQuadratureRule{RefQuadrilateral}(3)
F_ext = zeros(ndofs(dh)); rω = dof_range(dh, :ω)
for fc in FacetIterator(dh, getfacetset(grid, "right"))
    x, f = getcoordinates(fc), fc.current_facet_id; cd = celldofs(fc)
    for (ξ, w) in zip(fqr.facet_rules[f].points, fqr.facet_rules[f].weights)
        Jt = sum(Ferrite.reference_shape_gradient(ip, ξ, a)[f in (1, 3) ? 1 : 2] * x[a] for a in eachindex(x))
        for a in eachindex(x)
            F_ext[cd[rω[3(a - 1) + 2]]] -= M_ref / W * Ferrite.reference_shape_value(ip, ξ, a) * norm(Jt) * w
        end
    end
end
K, r, u = allocate_matrix(dh, ch), zeros(ndofs(dh)), zeros(ndofs(dh))
tipdofs = let tn = only(getnodeset(grid, "tip"))
    first((celldofs(c)[3a-2], celldofs(c)[3a]) for c in CellIterator(dh) for (a, n) in enumerate(getnodes(c)) if n == tn)
end
iters = Int[]; t_total = @elapsed for step in 1:nsteps
    λ = step / nsteps
    for it in 1:25
        assemble_global!(K, r, dh, s, Dn, u)
        rhs = λ * F_ext - r; apply_zero!(K, rhs, ch)
        norm(rhs) < 1e-6 && (push!(iters, it - 1); break)
        it == 25 && error("no convergence at step $step")
        du = K \ rhs; apply_zero!(du, ch); u .+= du
    end
    if step % (nsteps ÷ 4) == 0
        α = 2π * λ
        @printf("λ=%.2f  u_x=%9.5f (an %9.5f)  u_z=%9.5f (an %9.5f)\n", λ, u[tipdofs[1]], L*(sin(α)/α - 1), u[tipdofs[2]], L*(1 - cos(α))/α)
    end
end
@printf("Newton iterations/step: mean %.2f, max %d; total %.2f s\n", sum(iters)/length(iters), maximum(iters), t_total)
