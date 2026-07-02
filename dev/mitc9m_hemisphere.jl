using FerriteShells, LinearAlgebra, Printf

# Does MITC9M (membrane-tied) cure the pinched-hemisphere membrane locking?
# Linear solve, K assembled as the FD Hessian of energy_RM (so the membrane tying
# is active — the explicit membrane_tangent_RM! is still classical and would ignore it).
# Reference (P=1): |u_x(A)| = 0.0924.

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

function solve_hemisphere(n, mitc; mode=:fd)
    mat  = LinearElastic(6.825e7, 0.3, 0.04)
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
    K = allocate_matrix(dh); f = zeros(N)
    ke = zeros(5n_base, 5n_base); re = zeros(5n_base)
    asm = start_assemble(K, zeros(N))
    u0 = zeros(5n_base)
    for cell in CellIterator(dh)
        fill!(ke, 0.0)
        reinit!(scv, cell)
        if mode == :fd
            tangent_RM_FD!(ke, scv, u0, mat)            # FD Hessian → membrane tying active
        else
            membrane_tangent_RM!(ke, scv, u0, mat)      # explicit (classical membrane)
            bending_tangent_RM!(ke, scv, u0, mat)
        end
        assemble!(asm, shelldofs(cell), ke, re)
    end
    apply_pointload!(f, dh, "load_A", Vec{3}((-1.0, 0.0, 0.0)))
    apply_pointload!(f, dh, "load_B", Vec{3}(( 0.0, 1.0, 0.0)))
    apply!(K, f, ch)
    pd = isposdef(Symmetric(Matrix(K)))
    u_sol = K \ f
    ph = PointEvalHandler(grid, [grid.nodes[first(grid.nodesets["load_A"])].x])
    u_eval = first(evaluate_at_points(ph, dh, u_sol, :u))
    abs(u_eval[1]), pd
end

println("Pinched hemisphere — |u_x(A)| / $(U_REF)   [PD = stiffness pos.def.]")
@printf("%4s  %18s  %18s  %18s\n", "n", "MITC9 explicit", "MITC9 FD", "MITC9M FD")
for n in (4, 8, 16)
    ue, pe   = solve_hemisphere(n, MITC9;  mode=:explicit)
    uf, pf   = solve_hemisphere(n, MITC9;  mode=:fd)
    um, pm   = solve_hemisphere(n, MITC9M; mode=:fd)
    @printf("%4d  %6.4f(%3.0f%%,PD=%d)  %6.4f(%3.0f%%,PD=%d)  %6.4f(%3.0f%%,PD=%d)\n",
            n, ue,100ue/U_REF,pe, uf,100uf/U_REF,pf, um,100um/U_REF,pm)
end
