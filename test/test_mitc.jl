
@testset "MITC9 unit element" begin
    mat = LinearElastic(1.0e6, 0.3, 0.01)
    ip  = Lagrange{RefQuadrilateral, 2}()
    qr  = QuadratureRule{RefQuadrilateral}(3)
    scv_mitc   = ShellCellValues(qr, ip, ip; mitc=MITC9)
    scv_nomitc = ShellCellValues(qr, ip, ip)
    reinit!(scv_mitc,   X_Q9_UNIT)
    reinit!(scv_nomitc, X_Q9_UNIT)
    n_dof = 45

    # 1. All tying-point shear strains are zero at the reference state.
    γ₁_k, γ₂_k = FerriteShells.tying_shear_strains(scv_mitc.mitc, zeros(n_dof))
    @test all(v -> abs(v) ≤ 1e-14, γ₁_k)
    @test all(v -> abs(v) ≤ 1e-14, γ₂_k)

    # 2. Explicit residual matches ForwardDiff gradient (no-MITC path, exact to rounding).
    #    The no-MITC explicit formula is exact; MITC introduces a ~0.03% approximation
    #    in the shear-force variation (QP-direct δγ instead of MITC-interpolated δγ).
    Random.seed!(42)
    u_pert = zeros(n_dof)
    for I in 1:9
        u_pert[5I-2] = 1e-2 * sin(π * X_Q9_UNIT[I][1]) * sin(π * X_Q9_UNIT[I][2])
        u_pert[5I-1] = 1e-3 * randn()
        u_pert[5I  ] = 1e-3 * randn()
    end
    re_ex_nm = zeros(n_dof); bending_residuals_RM!(re_ex_nm, scv_nomitc, u_pert, mat)
    re_fd_nm = zeros(n_dof); bending_residuals_RM_FD!(re_fd_nm, scv_nomitc, u_pert, mat)
    @test norm(re_ex_nm .- re_fd_nm) / norm(re_fd_nm) < 1e-10

    # MITC explicit residual agrees with MITC ForwardDiff gradient to within the
    # approximation level (~0.1%); difference is the MITC-δγ consistency error.
    re_ex = zeros(n_dof); bending_residuals_RM!(re_ex, scv_mitc, u_pert, mat)
    re_fd = zeros(n_dof); bending_residuals_RM_FD!(re_fd, scv_mitc, u_pert, mat)
    @test norm(re_ex .- re_fd) / norm(re_fd) < 1e-2

    # Zero state.
    re_ex0 = zeros(n_dof); bending_residuals_RM!(re_ex0, scv_mitc, zeros(n_dof), mat)
    @test norm(re_ex0) ≤ 1e-14

    # Large rotations (~10°): same FD consistency checks.
    u_large = zeros(n_dof)
    α = deg2rad(10.0)
    for I in 1:9; u_large[5I-1] = α*0.3; u_large[5I] = α*0.7; end
    re_ex_lg_nm = zeros(n_dof); bending_residuals_RM!(re_ex_lg_nm, scv_nomitc, u_large, mat)
    re_fd_lg_nm = zeros(n_dof); bending_residuals_RM_FD!(re_fd_lg_nm, scv_nomitc, u_large, mat)
    @test norm(re_ex_lg_nm .- re_fd_lg_nm) / norm(re_fd_lg_nm) < 1e-10
    re_ex_lg = zeros(n_dof); bending_residuals_RM!(re_ex_lg, scv_mitc, u_large, mat)
    re_fd_lg = zeros(n_dof); bending_residuals_RM_FD!(re_fd_lg, scv_mitc, u_large, mat)
    @test norm(re_ex_lg .- re_fd_lg) / norm(re_fd_lg) < 1e-2

    # 2b. Consistent MITC9 tangent: bending_tangent_RM! (MITC dispatch) must match
    #     the ForwardDiff Jacobian of bending_residuals_RM! (not the energy Hessian —
    #     the MITC explicit residual is not the exact gradient of bending_shear_energy_RM).
    ke_ex  = zeros(n_dof, n_dof)
    bending_tangent_RM!(ke_ex, scv_mitc, u_pert, mat)
    ke_jac = ForwardDiff.jacobian(u -> begin
        re = zeros(eltype(u), n_dof)
        bending_residuals_RM!(re, scv_mitc, u, mat)
        re
    end, u_pert)
    @test norm(ke_ex .- ke_jac) / norm(ke_jac) < 1e-3

    # At zero state: tangent matches Jacobian of residual to near-machine precision.
    ke_ex0  = zeros(n_dof, n_dof)
    bending_tangent_RM!(ke_ex0, scv_mitc, zeros(n_dof), mat)
    ke_jac0 = ForwardDiff.jacobian(u -> begin
        re = zeros(eltype(u), n_dof)
        bending_residuals_RM!(re, scv_mitc, u, mat)
        re
    end, zeros(n_dof))
    @test norm(ke_ex0 .- ke_jac0) / norm(ke_jac0) < 1e-8

    # 3. Pure in-plane displacement on a flat element: d = G₃ = ẑ everywhere, so
    #    γ_α = a_α · ẑ = 0 — MITC and NoMITC must produce identical bending residuals.
    u_inplane = zeros(n_dof)
    for I in 1:9
        u_inplane[5I-4] = 1e-3 * X_Q9_UNIT[I][1]
        u_inplane[5I-3] = 2e-3 * X_Q9_UNIT[I][2]
    end
    re_mitc_ip   = zeros(n_dof); bending_residuals_RM!(re_mitc_ip,   scv_mitc,   u_inplane, mat)
    re_nomitc_ip = zeros(n_dof); bending_residuals_RM!(re_nomitc_ip, scv_nomitc, u_inplane, mat)
    @test re_mitc_ip ≈ re_nomitc_ip atol=1e-14

    # 4. MITC and NoMITC agree on the bending energy for the Kirchhoff mode
    #    u₃ = α·x·y, φ₁ = -α·y, φ₂ = -α·x  (zero transverse shear by construction).
    α_kl = 1e-4
    u_kl = zeros(n_dof)
    for I in 1:9
        xI, yI = X_Q9_UNIT[I][1], X_Q9_UNIT[I][2]
        u_kl[5I-2] = α_kl * xI * yI
        u_kl[5I-1] = -α_kl * yI
        u_kl[5I  ] = -α_kl * xI
    end
    W_mitc   = FerriteShells.bending_shear_energy_RM(u_kl, scv_mitc,   mat)
    W_nomitc = FerriteShells.bending_shear_energy_RM(u_kl, scv_nomitc, mat)
    @test W_mitc ≈ W_nomitc rtol=1e-6
end

@testset "MITC9 anti-locking: thin SS plate h-convergence" begin
    # Simply-supported square plate [0,1]² under sinusoidal load q₀·sin(πx)·sin(πy).
    # Same reference as the NoMITC h-convergence test in test_rm.jl.
    E, ν, t = 1e4, 0.3, 0.01
    D     = E * t^3 / (12 * (1 - ν^2))
    κ_s   = 5/6
    G_sh  = E / (2 * (1 + ν))
    q0    = 1.0
    w_nav = q0 / (4 * π^4 * D) + q0 / (2 * κ_s * G_sh * t * π^2)

    function ss_plate_center(n; use_mitc=false)
        ip  = Lagrange{RefQuadrilateral, 2}()
        qr  = QuadratureRule{RefQuadrilateral}(3)
        scv = ShellCellValues(qr, ip, ip; mitc = use_mitc ? MITC9 : nothing)
        mat_h = LinearElastic(E, ν, t)

        grid2d = generate_grid(QuadraticQuadrilateral, (n, n),
                               Vec{2}((0.0, 0.0)), Vec{2}((1.0, 1.0)))
        grid = shell_grid(grid2d)
        addnodeset!(grid, "boundary",
            x -> isapprox(x[1],0.0,atol=1e-10) || isapprox(x[1],1.0,atol=1e-10) ||
                 isapprox(x[2],0.0,atol=1e-10) || isapprox(x[2],1.0,atol=1e-10))

        dh = DofHandler(grid); add!(dh, :u, ip^3); add!(dh, :θ, ip^2); close!(dh)
        n_el   = ndofs_per_cell(dh)
        n_base = getnbasefunctions(ip)
        K = allocate_matrix(dh); f = zeros(ndofs(dh))
        asmb = start_assemble(K, zeros(ndofs(dh)))
        ke = zeros(n_el, n_el); re = zeros(n_el); fe = zeros(n_el)

        for cell in CellIterator(dh)
            fill!(ke, 0.0); fill!(re, 0.0); fill!(fe, 0.0)
            reinit!(scv, cell)
            x = getcoordinates(cell); u_e = zeros(n_el)
            membrane_tangent_RM!(ke, scv, u_e, mat_h)
            bending_tangent_RM_FD!(ke, scv, u_e, mat_h)
            assemble!(asmb, shelldofs(cell), ke, re)
            for qp in 1:getnquadpoints(scv)
                ξ  = scv.qr.points[qp]; dΩ = scv.detJdV[qp]
                xp = sum(Ferrite.reference_shape_value(ip, ξ, I) * x[I] for I in 1:n_base)
                q  = q0 * sin(π*xp[1]) * sin(π*xp[2])
                for I in 1:n_base
                    fe[5I-2] += Ferrite.reference_shape_value(ip, ξ, I) * q * dΩ
                end
            end
            @views f[shelldofs(cell)] .+= fe
        end

        dbc = ConstraintHandler(dh)
        add!(dbc, Dirichlet(:u, getnodeset(grid, "boundary"), x -> zeros(3), [1,2,3]))
        close!(dbc); Ferrite.update!(dbc, 0.0)
        apply!(K, f, dbc)
        u_sol = K \ f

        ph    = PointEvalHandler(grid, [Vec{3}((0.5, 0.5, 0.0))])
        return first(evaluate_at_points(ph, dh, u_sol, :u))[3]
    end

    # MITC9 convergence: use n=1,2,4 (solution essentially converges by n=4 for this
    # sinusoidal load; n=4→n=8 rates are not meaningful). Rates ≥ 1.5, <2% at n=4.
    ws_mitc = [ss_plate_center(n; use_mitc=true) for n in [1, 2, 4]]
    errors_mitc = abs.(ws_mitc .- w_nav)
    rates_mitc  = [log2(errors_mitc[i] / errors_mitc[i+1]) for i in 1:2]
    @test all(r -> r >= 1.5, rates_mitc)
    @test errors_mitc[end] / w_nav < 0.02

    # Anti-locking: at n=2, MITC9 must be strictly closer to the reference than
    # NoMITC, which suffers from shear locking.
    w_nomitc_2 = ss_plate_center(2; use_mitc=false)
    @test abs(ws_mitc[2] / w_nav - 1) < abs(w_nomitc_2 / w_nav - 1)
end


@testset "MITC4 unit element" begin
    mat = LinearElastic(1.0e6, 0.3, 0.01)
    ip  = Lagrange{RefQuadrilateral, 1}()
    qr  = QuadratureRule{RefQuadrilateral}(2)
    scv_mitc   = ShellCellValues(qr, ip, ip; mitc=MITC4)
    scv_nomitc = ShellCellValues(qr, ip, ip)
    reinit!(scv_mitc,   X_Q4_UNIT)
    reinit!(scv_nomitc, X_Q4_UNIT)
    n_dof = 20

    # 1. All tying-point shear strains are zero at the reference state.
    γ₁_k, γ₂_k = FerriteShells.tying_shear_strains(scv_mitc.mitc, zeros(n_dof))
    @test all(v -> abs(v) ≤ 1e-14, γ₁_k)
    @test all(v -> abs(v) ≤ 1e-14, γ₂_k)

    # 2. No-MITC explicit residual matches ForwardDiff gradient exactly.
    #    MITC4 explicit residual agrees to within the MITC-δγ approximation (~1%).
    Random.seed!(42)
    u_pert = zeros(n_dof)
    for I in 1:4
        u_pert[5I-2] = 1e-2 * sin(π * X_Q4_UNIT[I][1]) * sin(π * X_Q4_UNIT[I][2])
        u_pert[5I-1] = 1e-3 * randn()
        u_pert[5I  ] = 1e-3 * randn()
    end
    re_ex_nm = zeros(n_dof); bending_residuals_RM!(re_ex_nm, scv_nomitc, u_pert, mat)
    re_fd_nm = zeros(n_dof); bending_residuals_RM_FD!(re_fd_nm, scv_nomitc, u_pert, mat)
    @test norm(re_ex_nm .- re_fd_nm) / norm(re_fd_nm) < 1e-10

    re_ex = zeros(n_dof); bending_residuals_RM!(re_ex, scv_mitc, u_pert, mat)
    re_fd = zeros(n_dof); bending_residuals_RM_FD!(re_fd, scv_mitc, u_pert, mat)
    @test norm(re_ex .- re_fd) / norm(re_fd) < 1e-2

    # Zero state: residual is exactly zero.
    re_ex0 = zeros(n_dof); bending_residuals_RM!(re_ex0, scv_mitc, zeros(n_dof), mat)
    @test norm(re_ex0) ≤ 1e-14

    # 2b. Consistent MITC4 tangent: must match Jacobian of the explicit residual.
    ke_ex4  = zeros(n_dof, n_dof)
    bending_tangent_RM!(ke_ex4, scv_mitc, u_pert, mat)
    ke_jac4 = ForwardDiff.jacobian(u -> begin
        re = zeros(eltype(u), n_dof)
        bending_residuals_RM!(re, scv_mitc, u, mat)
        re
    end, u_pert)
    @test norm(ke_ex4 .- ke_jac4) / norm(ke_jac4) < 1e-3

    ke_ex40  = zeros(n_dof, n_dof)
    bending_tangent_RM!(ke_ex40, scv_mitc, zeros(n_dof), mat)
    ke_jac40 = ForwardDiff.jacobian(u -> begin
        re = zeros(eltype(u), n_dof)
        bending_residuals_RM!(re, scv_mitc, u, mat)
        re
    end, zeros(n_dof))
    @test norm(ke_ex40 .- ke_jac40) / norm(ke_jac40) < 1e-8

    # 3. Kirchhoff mode: u₃ = α·x·y, φ₁ = −α·y, φ₂ = −α·x (zero transverse shear).
    #    MITC4 and NoMITC must produce identical bending energy.
    α_kl = 1e-4
    u_kl = zeros(n_dof)
    for I in 1:4
        xI, yI = X_Q4_UNIT[I][1], X_Q4_UNIT[I][2]
        u_kl[5I-2] = α_kl * xI * yI
        u_kl[5I-1] = -α_kl * yI
        u_kl[5I  ] = -α_kl * xI
    end
    W_mitc   = FerriteShells.bending_shear_energy_RM(u_kl, scv_mitc,   mat)
    W_nomitc = FerriteShells.bending_shear_energy_RM(u_kl, scv_nomitc, mat)
    @test W_mitc ≈ W_nomitc rtol=1e-6
end

@testset "MITC4 anti-locking: thin SS plate h-convergence" begin
    E, ν, t = 1e4, 0.3, 0.01
    D     = E * t^3 / (12 * (1 - ν^2))
    κ_s   = 5/6
    G_sh  = E / (2 * (1 + ν))
    q0    = 1.0
    w_nav = q0 / (4 * π^4 * D) + q0 / (2 * κ_s * G_sh * t * π^2)

    function ss_plate_center_q4(n; use_mitc=false)
        ip  = Lagrange{RefQuadrilateral, 1}()
        qr  = QuadratureRule{RefQuadrilateral}(2)
        scv = ShellCellValues(qr, ip, ip; mitc = use_mitc ? MITC4 : nothing)
        mat_h = LinearElastic(E, ν, t)
        grid2d = generate_grid(Quadrilateral, (n, n),
                               Vec{2}((0.0, 0.0)), Vec{2}((1.0, 1.0)))
        grid = shell_grid(grid2d)
        addnodeset!(grid, "boundary",
            x -> isapprox(x[1],0.0,atol=1e-10) || isapprox(x[1],1.0,atol=1e-10) ||
                 isapprox(x[2],0.0,atol=1e-10) || isapprox(x[2],1.0,atol=1e-10))
        dh = DofHandler(grid); add!(dh, :u, ip^3); add!(dh, :θ, ip^2); close!(dh)
        n_el   = ndofs_per_cell(dh)
        n_base = getnbasefunctions(ip)
        K = allocate_matrix(dh); f = zeros(ndofs(dh))
        asmb = start_assemble(K, zeros(ndofs(dh)))
        ke = zeros(n_el, n_el); re = zeros(n_el); fe = zeros(n_el)
        for cell in CellIterator(dh)
            fill!(ke, 0.0); fill!(re, 0.0); fill!(fe, 0.0)
            reinit!(scv, cell)
            x = getcoordinates(cell); u_e = zeros(n_el)
            membrane_tangent_RM!(ke, scv, u_e, mat_h)
            bending_tangent_RM_FD!(ke, scv, u_e, mat_h)
            assemble!(asmb, shelldofs(cell), ke, re)
            for qp in 1:getnquadpoints(scv)
                ξ  = scv.qr.points[qp]; dΩ = scv.detJdV[qp]
                xp = sum(Ferrite.reference_shape_value(ip, ξ, I) * x[I] for I in 1:n_base)
                q  = q0 * sin(π*xp[1]) * sin(π*xp[2])
                for I in 1:n_base
                    fe[5I-2] += Ferrite.reference_shape_value(ip, ξ, I) * q * dΩ
                end
            end
            @views f[shelldofs(cell)] .+= fe
        end
        dbc = ConstraintHandler(dh)
        add!(dbc, Dirichlet(:u, getnodeset(grid, "boundary"), x -> zeros(3), [1,2,3]))
        close!(dbc); Ferrite.update!(dbc, 0.0)
        apply!(K, f, dbc)
        u_sol = K \ f
        ph = PointEvalHandler(grid, [Vec{3}((0.5, 0.5, 0.0))])
        return first(evaluate_at_points(ph, dh, u_sol, :u))[3]
    end

    # MITC4 convergence: n=2,4,8. Rates ≥ 1.5, error < 5% at n=8.
    ws_mitc = [ss_plate_center_q4(n; use_mitc=true) for n in [2, 4, 8]]
    errors_mitc = abs.(ws_mitc .- w_nav)
    rates_mitc  = [log2(errors_mitc[i] / errors_mitc[i+1]) for i in 1:2]
    @test all(r -> r >= 1.5, rates_mitc)
    @test errors_mitc[end] / w_nav < 0.05

    # Anti-locking: MITC4 at n=4 must be strictly closer to reference than no-MITC Q4.
    w_nomitc_4 = ss_plate_center_q4(4; use_mitc=false)
    @test abs(ws_mitc[2] / w_nav - 1) < abs(w_nomitc_4 / w_nav - 1)
end

@testset "MITC locking-free convergence: t/L sweep" begin
    # Simply-supported square plate [0,1]² under sinusoidal load q₀·sin(πx)·sin(πy).
    # Metric: L2 norm of the transverse displacement error against the Navier exact field
    #   w_exact(x,y) = W·sin(πx)·sin(πy),  W = q₀/(4π⁴D) + q₀/(2κGtπ²).
    # Using L2 rather than the center value avoids the super-convergence artefact (the
    # sinusoidal mode lies in the Q9 polynomial space, making the center deflection
    # converge to floating-point noise by n=8 for t/L=0.01).
    #
    # MITC9 convergence rates at t/L=0.001 (4 mesh levels, 3 rates): ≈ 3.0 (optimal for Q9).
    # MITC4 convergence rates at t/L=0.01 and 0.001 (4 mesh levels, 3 rates): ≈ 2.0 (optimal for Q4).
    # Without MITC, rates collapse to ≈ 0 for thin plates (locked stiffness, solution ≈ 0).
    E, ν, q0 = 1e4, 0.3, 1.0

    navier_ref(t) = let D = E*t^3 / (12*(1-ν^2)), G = E / (2*(1+ν))
        q0 / (4*π^4*D) + q0 / (2*(5/6)*G*t*π^2)
    end

    function ss_plate_l2err(n, t, ::Type{CT}, ip, qr; mitc_type=nothing) where CT
        W   = navier_ref(t)
        scv = ShellCellValues(qr, ip, ip; mitc=mitc_type)
        mat = LinearElastic(E, ν, t)
        n_base = getnbasefunctions(ip)
        grid = shell_grid(generate_grid(CT, (n, n), Vec{2}((0.,0.)), Vec{2}((1.,1.))))
        addnodeset!(grid, "boundary",
            x -> isapprox(x[1],0.,atol=1e-10) || isapprox(x[1],1.,atol=1e-10) ||
                 isapprox(x[2],0.,atol=1e-10) || isapprox(x[2],1.,atol=1e-10))
        dh = DofHandler(grid); add!(dh, :u, ip^3); add!(dh, :θ, ip^2); close!(dh)
        K = allocate_matrix(dh); f = zeros(ndofs(dh))
        asmb = start_assemble(K, zeros(ndofs(dh)))
        ke = zeros(5n_base, 5n_base); re = zeros(5n_base); fe = zeros(5n_base)
        for cell in CellIterator(dh)
            fill!(ke, 0.); fill!(re, 0.); fill!(fe, 0.)
            reinit!(scv, cell)
            x   = getcoordinates(cell); u_e = zeros(5n_base)
            membrane_tangent_RM!(ke, scv, u_e, mat)
            bending_tangent_RM_FD!(ke, scv, u_e, mat)
            assemble!(asmb, shelldofs(cell), ke, re)
            for qp in 1:getnquadpoints(scv)
                ξ = scv.qr.points[qp]; dΩ = scv.detJdV[qp]
                xp = sum(Ferrite.reference_shape_value(ip, ξ, I) * x[I] for I in 1:n_base)
                q  = q0 * sin(π*xp[1]) * sin(π*xp[2])
                for I in 1:n_base
                    fe[5I-2] += Ferrite.reference_shape_value(ip, ξ, I) * q * dΩ
                end
            end
            @views f[shelldofs(cell)] .+= fe
        end
        dbc = ConstraintHandler(dh)
        add!(dbc, Dirichlet(:u, getnodeset(grid, "boundary"), x -> zeros(3), [1,2,3]))
        close!(dbc); Ferrite.update!(dbc, 0.0); apply!(K, f, dbc)
        u_sol = K \ f
        err_sq = 0.0
        for cell in CellIterator(dh)
            reinit!(scv, cell)
            x   = getcoordinates(cell)
            u_e = u_sol[shelldofs(cell)]
            for qp in 1:getnquadpoints(scv)
                dΩ  = scv.detJdV[qp]
                xp  = Ferrite.spatial_coordinate(scv, qp, x)
                w_h = Ferrite.function_value(scv, qp, u_e)[3]
                err_sq += (w_h - W * sin(π*xp[1]) * sin(π*xp[2]))^2 * dΩ
            end
        end
        sqrt(err_sq)
    end

    # MITC9 (Q9, n=2,4,8,16): rates ≥ 2.5 at t/L=0.001.
    # At t/L=0.01 MITC9 reaches near-machine-precision error by n=8 (the element is
    # that accurate), so only the anti-locking comparison is checked there.
    let ip9 = Lagrange{RefQuadrilateral,2}(), qr9 = QuadratureRule{RefQuadrilateral}(4)
        errs_mitc = [ss_plate_l2err(n, 0.001, QuadraticQuadrilateral, ip9, qr9; mitc_type=MITC9) for n in [2, 4, 8, 16]]
        rates_mitc = [log2(errs_mitc[i] / errs_mitc[i+1]) for i in 1:3]
        @test all(r -> r >= 2.5, rates_mitc)

        for t in [0.01, 0.001]
            err_mitc   = ss_plate_l2err(4, t, QuadraticQuadrilateral, ip9, qr9; mitc_type=MITC9)
            err_nomitc = ss_plate_l2err(4, t, QuadraticQuadrilateral, ip9, qr9)
            @test err_mitc < err_nomitc / 5    # MITC9 must be at least 5× more accurate
        end
    end

    # MITC4 (Q4, n=4,8,16,32): rates ≥ 1.5 for both t/L=0.01 and t/L=0.001.
    # No-MITC Q4 is almost entirely locked (error ≈ solution magnitude) for both thicknesses.
    let ip4 = Lagrange{RefQuadrilateral,1}(), qr4 = QuadratureRule{RefQuadrilateral}(3)
        for t in [0.01, 0.001]
            errs_mitc   = [ss_plate_l2err(n, t, Quadrilateral, ip4, qr4; mitc_type=MITC4) for n in [4, 8, 16, 32]]
            errs_nomitc = [ss_plate_l2err(n, t, Quadrilateral, ip4, qr4) for n in [4, 8, 16, 32]]
            rates_mitc  = [log2(errs_mitc[i] / errs_mitc[i+1]) for i in 1:3]
            @test all(r -> r >= 1.5, rates_mitc)
            @test errs_mitc[end] < errs_nomitc[end] / 5   # MITC4 at least 5× more accurate at finest mesh
        end
    end
end
