using FerriteShells, LinearAlgebra, Printf

const U_REF = 0.0924

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

function solve_hemisphere(n, mitc, t; use_nf=false)
    mat  = LinearElastic(6.825e7, 0.3, t)
    grid = hemisphere_grid(n)
    ip   = Lagrange{RefQuadrilateral, 2}()
    qr   = QuadratureRule{RefQuadrilateral}(3)
    scv  = ShellCellValues(qr, ip, ip; mitc=mitc)
    nf   = NodeFrames(grid, ip)
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
        fill!(ke, 0.0)
        use_nf ? reinit!(scv, cell, nf) : reinit!(scv, cell)
        tangent_RM_FD!(ke, scv, u0, mat)
        assemble!(asm, shelldofs(cell), ke, re)
    end
    apply_pointload!(f, dh, "load_A", Vec{3}((-1.0,0.0,0.0)))
    apply_pointload!(f, dh, "load_B", Vec{3}(( 0.0,1.0,0.0)))
    free = setdiff(1:N, ch.prescribed_dofs)
    nneg = count(<(0), eigvals(Symmetric(Matrix(K[free,free]))))
    apply!(K, f, ch)
    u = K \ f
    ph = PointEvalHandler(grid, [grid.nodes[first(grid.nodesets["load_A"])].x])
    ux = abs(first(evaluate_at_points(ph, dh, u, :u))[1])
    ux, nneg
end

println("Thin hemisphere t=0.04 — |u_x(A)|/$(U_REF), neg-eigs   (MITC9)")
for nf in (false, true)
    for n in (8, 16)
        ux, nneg = solve_hemisphere(n, MITC9, 0.04; use_nf=nf)
        @printf("  nf=%-5s n=%-3d  %6.4f (%3.0f%%)  neg=%d\n", nf, n, ux, 100ux/U_REF, nneg)
    end
end

println("\nThin hemisphere t=0.04 with NodeFrames — MITC9 vs MITC9M")
for n in (8, 16, 32)
    u9, ng9   = solve_hemisphere(n, MITC9,  0.04; use_nf=true)
    um, ngm   = solve_hemisphere(n, MITC9M, 0.04; use_nf=true)
    @printf("  n=%-3d  MITC9 %6.4f(%3.0f%%,neg=%d)   MITC9M %6.4f(%3.0f%%,neg=%d)\n",
            n, u9,100u9/U_REF,ng9, um,100um/U_REF,ngm)
end
