using FerriteShells, LinearAlgebra, Printf

function hemisphere_grid(n; R=10.0, θ_hole_deg=18.0)
    θ_min = θ_hole_deg * π / 180
    g = shell_grid(
        generate_grid(QuadraticQuadrilateral, (n, n), Vec{2}((θ_min, 0.0)), Vec{2}((π/2, π/2)));
        map = nd -> (R*sin(nd.x[1])*cos(nd.x[2]), R*sin(nd.x[1])*sin(nd.x[2]), R*cos(nd.x[1])))
    addfacetset!(g, "sym_phi0",  x -> abs(x[2]) < 1e-10)
    addfacetset!(g, "sym_phi90", x -> abs(x[1]) < 1e-10)
    addnodeset!(g, "load_A", x -> abs(x[3]) < 1e-6 && abs(x[2]) < 1e-6 && x[1] > 0.5R)
    addnodeset!(g, "load_B", x -> abs(x[3]) < 1e-6 && abs(x[1]) < 1e-6 && x[2] > 0.5R)
    return g
end

function assemble(n, mitc, t)
    mat  = LinearElastic(6.825e7, 0.3, t)
    grid = hemisphere_grid(n)
    ip   = Lagrange{RefQuadrilateral, 2}()
    qr   = QuadratureRule{RefQuadrilateral}(3)
    scv  = ShellCellValues(qr, ip, ip; mitc=mitc)
    dh = DofHandler(grid); add!(dh, :u, ip^3); add!(dh, :θ, ip^2); close!(dh)
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "sym_phi0"),  x -> 0.0, [2]))
    add!(ch, Dirichlet(:u, getfacetset(grid, "sym_phi90"), x -> 0.0, [1]))
    add!(ch, Dirichlet(:θ, getfacetset(grid, "sym_phi0"),  x -> 0.0, [2]))
    add!(ch, Dirichlet(:θ, getfacetset(grid, "sym_phi90"), x -> 0.0, [2]))
    close!(ch); Ferrite.update!(ch, 0.0)
    N = ndofs(dh); n_base = getnbasefunctions(ip)
    K = allocate_matrix(dh); f = zeros(N); ke = zeros(5n_base,5n_base); re = zeros(5n_base)
    asm = start_assemble(K, zeros(N)); u0 = zeros(5n_base)
    for cell in CellIterator(dh)
        fill!(ke, 0.0); reinit!(scv, cell)
        tangent_RM_FD!(ke, scv, u0, mat)
        assemble!(asm, shelldofs(cell), ke, re)
    end
    apply_pointload!(f, dh, "load_A", Vec{3}((-1.0,0.0,0.0)))
    apply_pointload!(f, dh, "load_B", Vec{3}(( 0.0,1.0,0.0)))
    apply!(K, f, ch)
    u = K \ f
    ph = PointEvalHandler(grid, [grid.nodes[first(grid.nodesets["load_A"])].x])
    ux = abs(first(evaluate_at_points(ph, dh, u, :u))[1])
    K, ux, ch
end

# free-DOF negative eigenvalue count (drop constrained rows/cols)
function neg_count(K, ch)
    free = setdiff(1:size(K,1), ch.prescribed_dofs)
    λ = eigvals(Symmetric(Matrix(K[free, free])))
    count(<(-1e-6*maximum(abs,λ)), λ), length(free)
end

println("=== indefiniteness of hemisphere stiffness (thin, t=0.04) ===")
for mitc in (MITC9, MITC9M)
    K, ux, ch = assemble(8, mitc, 0.04)
    nneg, nfree = neg_count(K, ch)
    @printf("n=8  %-7s  neg-eigs=%d / %d free   |u_x(A)|=%.4g\n",
            mitc===MITC9 ? "MITC9" : "MITC9M", nneg, nfree, ux)
end

println("\n=== thick hemisphere t=0.4 (well-conditioned, membrane action) ===")
println("converged value comparison MITC9 vs MITC9M (self-convergence):")
for n in (4, 8, 16)
    _, u9,  _ = assemble(n, MITC9,  0.4)
    _, u9m, _ = assemble(n, MITC9M, 0.4)
    @printf("n=%-3d  MITC9=%.5g   MITC9M=%.5g   ratio=%.3f\n", n, u9, u9m, u9m/u9)
end
