using FerriteShells
using LinearAlgebra
using Test

# Scordelis-Lo roof (RM)
function scordelis_lo_rm_solve_test(ns, nt)
    R_sl, L_sl, Φ_sl = 25.0, 50.0, 40π/180
    E_sl, ν_sl, t_sl = 4.32e8, 0.0, 0.25
    q_sl = Vec{3}((0.0, -90.0, 0.0))

    ip  = Lagrange{RefQuadrilateral, 2}()
    qr  = QuadratureRule{RefQuadrilateral}(4)
    scv = ShellCellValues(qr, ip, ip)
    mat = LinearElastic(E_sl, ν_sl, t_sl)

    grid = shell_grid(
        generate_grid(QuadraticQuadrilateral, (ns, nt),
                      Vec{2}((-Φ_sl, 0.0)), Vec{2}((Φ_sl, L_sl)));
        map = n -> (n.x[2], R_sl * cos(n.x[1]), R_sl * sin(n.x[1])))
    addnodeset!(grid, "diaphragm", x -> x[1] ≈ 0.0 || x[1] ≈ L_sl)
    addnodeset!(grid, "ref_point",
        x -> abs(x[1] - L_sl/2) < 1e-8 && abs(x[2] - R_sl*cos(Φ_sl)) < 1e-8 &&
             abs(x[3] - R_sl*sin(Φ_sl)) < 1e-8)

    dh = DofHandler(grid); add!(dh, :u, ip^3); add!(dh, :θ, ip^2); close!(dh)
    n_base = getnbasefunctions(ip)

    K  = allocate_matrix(dh)
    f  = zeros(ndofs(dh))
    asmb = start_assemble(K, zeros(ndofs(dh)))
    ke = zeros(5n_base, 5n_base); re = zeros(5n_base); fe = zeros(5n_base)

    for cell in CellIterator(dh)
        fill!(ke, 0.0); fill!(re, 0.0); fill!(fe, 0.0)
        reinit!(scv, cell)
        x  = getcoordinates(cell)
        u0 = zeros(5n_base)
        tangent_RM_FD!(ke, scv, u0, mat)
        sd = shelldofs(cell)
        assemble!(asmb, sd, ke, re)
        for qp in 1:getnquadpoints(scv)
            ξ  = scv.qr.points[qp]; dΩ = scv.detJdV[qp]
            for I in 1:n_base
                NI = Ferrite.reference_shape_value(ip, ξ, I)
                @views fe[5I-4:5I-2] .+= NI * q_sl * dΩ
            end
        end
        @views f[sd] .+= fe
    end

    dbc = ConstraintHandler(dh)
    add!(dbc, Dirichlet(:u, getnodeset(grid, "diaphragm"), x -> zeros(2), [2, 3]))
    close!(dbc); Ferrite.update!(dbc, 0.0); apply!(K, f, dbc)
    u_sol = K \ f

    ref_nodes = collect(getnodeset(grid, "ref_point"))
    for cell in CellIterator(dh)
        for (I, gid) in enumerate(getnodes(cell))
            if gid == ref_nodes[1]
                cd = celldofs(cell)
                return u_sol[cd[3I-1]]
            end
        end
    end
    error("ref_point not found")
end

# Pinched cylinder (RM, 1/8 symmetry)
function pinched_cylinder_rm_solve_test(ns, na)
    R_pc, L_pc = 300.0, 600.0
    E_pc, ν_pc, t_pc = 3.0e6, 0.3, 3.0
    P_pc = 1.0

    ip  = Lagrange{RefQuadrilateral, 2}()
    qr  = QuadratureRule{RefQuadrilateral}(4)
    scv = ShellCellValues(qr, ip, ip)
    mat = LinearElastic(E_pc, ν_pc, t_pc)

    grid = shell_grid(
        generate_grid(QuadraticQuadrilateral, (ns, na),
                      Vec{2}((0.0, 0.0)), Vec{2}((π/2, L_pc/2)));
        map = n -> (n.x[2], R_pc * sin(n.x[1]), R_pc * cos(n.x[1])))
    addnodeset!(grid, "diaphragm",   x -> x[1] ≈ 0.0)
    addnodeset!(grid, "sym_axial",   x -> x[1] ≈ L_pc/2)
    addnodeset!(grid, "sym_theta0",  x -> abs(x[2]) < 1e-6)
    addnodeset!(grid, "sym_theta90", x -> abs(x[3]) < 1e-6)
    addnodeset!(grid, "load_point",
        x -> x[1] ≈ L_pc/2 && abs(x[2]) < 1e-6 && abs(x[3] - R_pc) < 1e-6)

    dh = DofHandler(grid); add!(dh, :u, ip^3); add!(dh, :θ, ip^2); close!(dh)
    n_base = getnbasefunctions(ip)

    K  = allocate_matrix(dh)
    f  = zeros(ndofs(dh))
    asmb = start_assemble(K, zeros(ndofs(dh)))
    ke = zeros(5n_base, 5n_base); re = zeros(5n_base)

    for cell in CellIterator(dh)
        fill!(ke, 0.0); fill!(re, 0.0)
        reinit!(scv, cell)
        x  = getcoordinates(cell)
        u0 = zeros(5n_base)
        tangent_RM_FD!(ke, scv, u0, mat)
        assemble!(asmb, shelldofs(cell), ke, re)
    end

    apply_pointload!(f, dh, "load_point", Vec{3}((0.0, 0.0, -P_pc / 4)))

    dbc = ConstraintHandler(dh)
    add!(dbc, Dirichlet(:u, getnodeset(grid, "diaphragm"),   x -> zeros(2), [2, 3]))
    add!(dbc, Dirichlet(:u, getnodeset(grid, "sym_axial"),   x -> 0.0,      [1]))
    add!(dbc, Dirichlet(:u, getnodeset(grid, "sym_theta0"),  x -> 0.0,      [2]))
    add!(dbc, Dirichlet(:u, getnodeset(grid, "sym_theta90"), x -> 0.0,      [3]))
    add!(dbc, Dirichlet(:θ, getnodeset(grid, "sym_theta0"),  x -> 0.0, [2]))
    add!(dbc, Dirichlet(:θ, getnodeset(grid, "sym_theta90"), x -> 0.0, [2]))
    add!(dbc, Dirichlet(:θ, getnodeset(grid, "sym_axial"),   x -> 0.0, [1]))
    close!(dbc); Ferrite.update!(dbc, 0.0); apply!(K, f, dbc)
    u_sol = K \ f

    load_nodes = collect(getnodeset(grid, "load_point"))
    for cell in CellIterator(dh)
        for (I, gid) in enumerate(getnodes(cell))
            if gid == load_nodes[1]
                cd = celldofs(cell)
                return u_sol[cd[3I]]
            end
        end
    end
    error("load_point not found")
end

# Tests
@testset "Scordelis-Lo roof (RM) h-convergence" begin
    ref = -0.3024
    ws  = [scordelis_lo_rm_solve_test(n, n) for n in [4, 8, 16]]
    errs = abs.(ws .- ref)
    rates = [log2(errs[i] / errs[i+1]) for i in 1:length(errs)-1]
    @test all(r -> r >= 1.5, rates)
    @test errs[end] / abs(ref) < 0.05
end

@testset "Pinched cylinder (RM) h-convergence" begin
    ref = -1.8248e-5
    ws  = [pinched_cylinder_rm_solve_test(n, n) for n in [8, 16]]
    errs = abs.(ws .- ref)
    @test errs[1] > errs[2]              # monotone convergence
    @test errs[2] / abs(ref) < 0.12     # 16×16 within 12% of reference
end

# Pinched hemisphere (RM, quarter symmetry, t/R = 0.004)
# R = 10, t = 0.04, E = 6.825e7, ν = 0.3; P = 1 inward at A = (R,0,0), outward at B = (0,R,0).
# Reference (linear): |u_x(A)| = 0.0924.
# Needs all three of: MITC (bending-dominated), NodeFrames (per-node frame so the φ DOFs
# have a single meaning), and the frame-independent director symmetry BC. Writing the
# symmetry condition as `Dirichlet(:θ, set, x -> 0.0, [2])` clamps the shell at the
# equator — where the frame heuristic flips and the load sits — and stalls at 99% error.
function pinched_hemisphere_rm_solve_test(n)
    R, θ_min = 10.0, 18π/180
    mat = LinearElastic(6.825e7, 0.3, 0.04)
    grid = shell_grid(
        generate_grid(QuadraticQuadrilateral, (n, n), Vec{2}((θ_min, 0.0)), Vec{2}((π/2, π/2)));
        map = nd -> (R*sin(nd.x[1])*cos(nd.x[2]), R*sin(nd.x[1])*sin(nd.x[2]), R*cos(nd.x[1])))
    addfacetset!(grid, "sym_phi0",  x -> abs(x[2]) < 1e-10)
    addfacetset!(grid, "sym_phi90", x -> abs(x[1]) < 1e-10)
    addnodeset!(grid, "sym_phi0_n",  x -> abs(x[2]) < 1e-9)
    addnodeset!(grid, "sym_phi90_n", x -> abs(x[1]) < 1e-9)
    addnodeset!(grid, "load_A", x -> abs(x[3]) < 1e-6 && abs(x[2]) < 1e-6 && x[1] > 0.5R)
    addnodeset!(grid, "load_B", x -> abs(x[3]) < 1e-6 && abs(x[1]) < 1e-6 && x[2] > 0.5R)

    ip  = Lagrange{RefQuadrilateral, 2}()
    scv = ShellCellValues(QuadratureRule{RefQuadrilateral}(3), ip, ip; mitc=MITC9)
    nf  = NodeFrames(grid, ip)
    dh  = DofHandler(grid); add!(dh, :u, ip^3); add!(dh, :θ, ip^2); close!(dh)

    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "sym_phi0"),  x -> 0.0, [2]))
    add!(ch, Dirichlet(:u, getfacetset(grid, "sym_phi90"), x -> 0.0, [1]))
    add_director_symmetry!(ch, dh, nf, "sym_phi0_n",  Vec{3}((0.0, 1.0, 0.0)))
    add_director_symmetry!(ch, dh, nf, "sym_phi90_n", Vec{3}((1.0, 0.0, 0.0)))
    close!(ch); Ferrite.update!(ch, 0.0)

    n_base = getnbasefunctions(ip)
    K = allocate_matrix(dh, ch); f = zeros(ndofs(dh))
    ke = zeros(5n_base, 5n_base); re = zeros(5n_base)
    asm = start_assemble(K, zeros(ndofs(dh)))
    for cell in CellIterator(dh)
        fill!(ke, 0.0)
        reinit!(scv, cell, nf)
        u0 = zeros(5n_base)
        membrane_tangent_RM!(ke, scv, u0, mat)
        bending_tangent_RM!(ke, scv, u0, mat)
        assemble!(asm, shelldofs(cell), ke, re)
    end
    apply_pointload!(f, dh, "load_A", Vec{3}((-1.0, 0.0, 0.0)))
    apply_pointload!(f, dh, "load_B", Vec{3}(( 0.0, 1.0, 0.0)))
    apply!(K, f, ch)
    u = K \ f
    apply!(u, ch)

    nid = first(getnodeset(grid, "load_A"))
    for cell in CellIterator(dh)
        sd = shelldofs(cell)
        for (I, g) in enumerate(getnodes(cell))
            g == nid && return u[sd[5I-4]]
        end
    end
    error("load_A not found")
end

@testset "Pinched hemisphere (RM) h-convergence" begin
    ref = -0.0924
    us   = [pinched_hemisphere_rm_solve_test(n) for n in (8, 16, 32)]
    errs = abs.(us .- ref)
    @test all(diff(errs) .< 0)              # monotone convergence
    @test errs[end] / abs(ref) < 0.02       # 32×32 within 2% of the reference
end
