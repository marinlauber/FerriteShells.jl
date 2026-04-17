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
        membrane_tangent_RM_FD!(ke, scv, u0, mat)
        bending_tangent_RM_FD!(ke, scv, u0, mat)
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
        membrane_tangent_RM_FD!(ke, scv, u0, mat)
        bending_tangent_RM_FD!(ke, scv, u0, mat)
        assemble!(asmb, shelldofs(cell), ke, re)
    end

    apply_pointload!(f, dh, "load_point", Vec{3}((0.0, 0.0, -P_pc / 4)))

    dbc = ConstraintHandler(dh)
    add!(dbc, Dirichlet(:u, getnodeset(grid, "diaphragm"),   x -> zeros(2), [2, 3]))
    add!(dbc, Dirichlet(:u, getnodeset(grid, "sym_axial"),   x -> 0.0,      [1]))
    add!(dbc, Dirichlet(:u, getnodeset(grid, "sym_theta0"),  x -> 0.0,      [2]))
    add!(dbc, Dirichlet(:u, getnodeset(grid, "sym_theta90"), x -> 0.0,      [3]))
    add!(dbc, Dirichlet(:θ, getnodeset(grid, "sym_theta0"),  x -> 0.0, [1]))
    add!(dbc, Dirichlet(:θ, getnodeset(grid, "sym_theta90"), x -> 0.0, [1]))
    add!(dbc, Dirichlet(:θ, getnodeset(grid, "sym_axial"),   x -> 0.0, [2]))
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
