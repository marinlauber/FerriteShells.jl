
@testset "triangular MITC assumed-strain interpolation" begin
    # The builder must reproduce every field of its own assumed space exactly: sampling a
    # basis field at the tying entries and interpolating back to the quadrature points
    # returns the field itself. Catches a wrong tying point, a singular tying matrix, or a
    # basis/condition mismatch.
    qr = QuadratureRule{RefTriangle}(4)
    for (mitc_ctor, M) in ((MITC3, 4), (MITC6a, 10))
        conds, basis = FerriteShells.tying_conditions(mitc_ctor)
        ξ_tie, α_tie, h₁, h₂ = FerriteShells.tying_weights(qr, conds, basis)
        @test length(ξ_tie) == M
        for P in basis
            γ = [P(ξ_tie[k])[α_tie[k]] for k in eachindex(ξ_tie)]
            for q in eachindex(qr.weights)
                @test sum(h₁[q,:] .* γ) ≈ P(qr.points[q])[1] atol=1e-12
                @test sum(h₂[q,:] .* γ) ≈ P(qr.points[q])[2] atol=1e-12
            end
        end
    end

    # MITC3 against the closed form of Lee & Bathe Eq. (25):
    #   γ̃₁ = γ¹ + c·s, γ̃₂ = γ² − c·r  with  c = γ² − γ¹ + γ³ − γ⁴,
    # entries ordered (A,1), (B,2), (C,1), (C,2).
    conds, basis = FerriteShells.tying_conditions(MITC3)
    _, _, h₁, h₂ = FerriteShells.tying_weights(qr, conds, basis)
    for q in eachindex(qr.weights)
        r, s = qr.points[q][1], qr.points[q][2]
        @test h₁[q,:] ≈ [1-s, s, s, -s] atol=1e-14
        @test h₂[q,:] ≈ [r, 1-r, -r, r] atol=1e-14
    end
end

@testset "triangular MITC unit element" begin
    mat = LinearElastic(1.0e6, 0.3, 0.01)
    # Pure-bending energy of the twist mode u₃ = α·x·y, φ₁ = −α·y, φ₂ = −α·x on the unit
    # triangle: the mode lies in the T6 space and is shear-free there, so the displacement-based
    # T6 already gives the exact value. Used as the reference for both elements below.
    α = 1e-4
    W_twist = let ip = Lagrange{RefTriangle,2}(), qr = QuadratureRule{RefTriangle}(4)
        scv = ShellCellValues(qr, ip, ip); reinit!(scv, X_T6_UNIT)
        u = zeros(30)
        for I in 1:6
            xI, yI = X_T6_UNIT[I][1], X_T6_UNIT[I][2]
            u[5I-2] = α*xI*yI; u[5I-1] = -α*yI; u[5I] = -α*xI
        end
        FerriteShells.energy_RM(u, scv, mat)
    end

    for (mitc_ctor, ip, x_nodes) in ((MITC3,  Lagrange{RefTriangle,1}(), X_T3_UNIT),
                                     (MITC6a, Lagrange{RefTriangle,2}(), X_T6_UNIT))
        qr = QuadratureRule{RefTriangle}(4)
        scv_mitc   = ShellCellValues(qr, ip, ip; mitc=mitc_ctor)
        scv_nomitc = ShellCellValues(qr, ip, ip)
        reinit!(scv_mitc, x_nodes); reinit!(scv_nomitc, x_nodes)
        n_base = getnbasefunctions(ip); n_dof = 5n_base

        # 1. Reference state is strain- and stress-free.
        γ_k = FerriteShells.tying_shear_strains(scv_mitc.mitc, zeros(n_dof))
        @test all(v -> abs(v) ≤ 1e-14, γ_k)
        re0 = zeros(n_dof); bending_residuals_RM!(re0, scv_mitc, zeros(n_dof), mat)
        @test norm(re0) ≤ 1e-14

        # 2. Explicit residual/tangent are the exact gradient/Hessian of `energy_RM` — the
        #    MITC B-operators must differentiate through the tying interpolation.
        Random.seed!(42)
        u_pert = zeros(n_dof)
        for I in 1:n_base
            u_pert[5I-2] = 1e-2 * sin(π*x_nodes[I][1]) * sin(π*x_nodes[I][2])
            u_pert[5I-1] = 1e-3 * randn(); u_pert[5I] = 1e-3 * randn()
        end
        re_ex  = zeros(n_dof); bending_residuals_RM!(re_ex, scv_mitc, u_pert, mat)
        re_fd  = zeros(n_dof); residuals_RM_FD!(re_fd, scv_mitc, u_pert, mat)
        re_mem = zeros(n_dof); membrane_residuals_RM!(re_mem, scv_mitc, u_pert, mat)
        @test norm(re_ex .- (re_fd .- re_mem)) / norm(re_ex) < 1e-10
        ke_ex  = zeros(n_dof, n_dof); bending_tangent_RM!(ke_ex, scv_mitc, u_pert, mat)
        ke_jac = ForwardDiff.jacobian(u -> begin
            re = zeros(eltype(u), n_dof)
            bending_residuals_RM!(re, scv_mitc, u, mat)
            re
        end, u_pert)
        @test norm(ke_ex .- ke_jac) / norm(ke_jac) < 1e-8
        @test norm(ke_ex .- ke_ex') / norm(ke_ex) < 1e-10

        # 3. Pure in-plane displacement on a flat element: d = G₃ ⇒ γ_α = 0, MITC ≡ NoMITC.
        u_ip = zeros(n_dof)
        for I in 1:n_base
            u_ip[5I-4] = 1e-3 * x_nodes[I][1]; u_ip[5I-3] = 2e-3 * x_nodes[I][2]
        end
        re_m = zeros(n_dof); bending_residuals_RM!(re_m, scv_mitc,   u_ip, mat)
        re_n = zeros(n_dof); bending_residuals_RM!(re_n, scv_nomitc, u_ip, mat)
        @test re_m ≈ re_n atol=1e-14

        # 4. Kirchhoff twist mode: the tied element must return the pure-bending energy.
        #    On T6 the mode is in the shape-function space and is shear-free pointwise, so
        #    tying changes nothing. On T3 it is not representable — the displacement-based
        #    element then sees spurious shear (locking) that the tying removes, so MITC3
        #    recovers the same bending energy while NoMITC overshoots by orders of magnitude.
        u_kl = zeros(n_dof)
        for I in 1:n_base
            xI, yI = x_nodes[I][1], x_nodes[I][2]
            u_kl[5I-2] = α*xI*yI; u_kl[5I-1] = -α*yI; u_kl[5I] = -α*xI
        end
        W_mitc   = FerriteShells.energy_RM(u_kl, scv_mitc,   mat)
        W_nomitc = FerriteShells.energy_RM(u_kl, scv_nomitc, mat)
        @test W_mitc ≈ W_twist rtol=1e-3
        if n_base == 6
            @test W_mitc ≈ W_nomitc rtol=1e-6
        else
            @test W_nomitc > 100 * W_mitc
        end
    end
end

@testset "triangular MITC spatial isotropy" begin
    # The Lee & Bathe construction exists precisely so the tied strain field does not depend
    # on which corner is numbered first — a scheme that treats the two natural directions
    # asymmetrically fails here. Cyclic renumbering must permute the element tangent, not
    # change it. Checked on a distorted triangle, where an anisotropic scheme would differ.
    mat = LinearElastic(1.0e6, 0.3, 0.01)
    x_t3 = [Vec{3}((0.0,0.0,0.0)), Vec{3}((1.0,0.2,0.0)), Vec{3}((0.1,0.9,0.0))]
    x_t6 = [x_t3; [(x_t3[1]+x_t3[2])/2, (x_t3[2]+x_t3[3])/2, (x_t3[3]+x_t3[1])/2]]
    for (mitc_ctor, ip, x_nodes, cyc) in ((MITC3,  Lagrange{RefTriangle,1}(), x_t3, [2,3,1]),
                                          (MITC6a, Lagrange{RefTriangle,2}(), x_t6, [2,3,1,5,6,4]))
        qr  = QuadratureRule{RefTriangle}(4)
        scv = ShellCellValues(qr, ip, ip; mitc=mitc_ctor)
        n_dof = 5 * getnbasefunctions(ip)
        perm  = vcat([[5c-4, 5c-3, 5c-2, 5c-1, 5c] for c in cyc]...)

        reinit!(scv, x_nodes)
        ke = zeros(n_dof, n_dof); tangent_RM_FD!(ke, scv, zeros(n_dof), mat)
        reinit!(scv, x_nodes[cyc])
        ke_c = zeros(n_dof, n_dof); tangent_RM_FD!(ke_c, scv, zeros(n_dof), mat)
        @test norm(ke_c .- ke[perm, perm]) / norm(ke) < 1e-10
    end
end

@testset "triangular MITC reference state is positive semidefinite" begin
    # Guards spurious zero-energy modes admitted by the enlarged assumed-strain space and a
    # pre-stressed reference on curved elements (cf. the MITC9 double reference-shear bug).
    mat = LinearElastic(1.0e6, 0.3, 0.01)
    warp(p) = Vec{3}((p[1], p[2], 0.15*(p[1]^2 + p[2]^2)))
    for (mitc_ctor, ip, x_nodes) in ((MITC3,  Lagrange{RefTriangle,1}(), X_T3_UNIT),
                                     (MITC6a, Lagrange{RefTriangle,2}(), X_T6_UNIT))
        qr    = QuadratureRule{RefTriangle}(4)
        scv   = ShellCellValues(qr, ip, ip; mitc=mitc_ctor)
        n_dof = 5 * getnbasefunctions(ip)
        for X in (x_nodes, map(warp, x_nodes))   # a T3 stays flat; the warped T6 is curved
            reinit!(scv, X)
            ke = zeros(n_dof, n_dof); tangent_RM_FD!(ke, scv, zeros(n_dof), mat)
            λ  = eigvals(Symmetric(ke)); tol = 1e-7 * maximum(abs, λ)
            @test count(<(-tol), λ) == 0
            @test count(v -> abs(v) ≤ tol, λ) == 6
        end
    end
end

@testset "triangular MITC rigid-body rotation" begin
    # Finite rigid rotation about the y-axis leaves no residual, single element and patch.
    mat = LinearElastic(1.0e6, 0.3, 0.1)
    α   = deg2rad(5.0)
    x_t3 = [Vec{3}((0.0,0.0,0.0)), Vec{3}((1.0,0.2,0.0)), Vec{3}((0.1,0.9,0.0))]
    x_t6 = [x_t3; [(x_t3[1]+x_t3[2])/2, (x_t3[2]+x_t3[3])/2, (x_t3[3]+x_t3[1])/2]]
    for (mitc_ctor, ip, x_nodes) in ((MITC3,  Lagrange{RefTriangle,1}(), x_t3),
                                     (MITC6a, Lagrange{RefTriangle,2}(), x_t6))
        qr  = QuadratureRule{RefTriangle}(4)
        scv = ShellCellValues(qr, ip, ip; mitc=mitc_ctor)
        reinit!(scv, x_nodes)
        n_dof = 5 * getnbasefunctions(ip); u_e = zeros(n_dof)
        for I in eachindex(x_nodes)
            u_e[5I-4] = x_nodes[I][1] * (cos(α) - 1)
            u_e[5I-2] = x_nodes[I][1] * sin(α)
            u_e[5I-1] = -α
        end
        re = zeros(n_dof)
        membrane_residuals_RM!(re, scv, u_e, mat)
        bending_residuals_RM!(re, scv, u_e, mat)
        @test norm(re) < 1e-8
    end
end

@testset "triangular MITC anti-locking: thin SS plate h-convergence" begin
    # Simply-supported [0,1]² plate under q₀·sin(πx)·sin(πy); L2 error of w against the
    # Navier solution, same metric as the quadrilateral t/L sweep.
    E, ν, q0 = 1e4, 0.3, 1.0
    navier_ref(t) = let D = E*t^3 / (12*(1-ν^2)), G = E / (2*(1+ν))
        q0 / (4*π^4*D) + q0 / (2*(5/6)*G*t*π^2)
    end

    function ss_plate_l2err_tri(n, t, ::Type{CT}, ip, qr; mitc_type=nothing) where CT
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
            x = getcoordinates(cell)
            tangent_RM_FD!(ke, scv, zeros(5n_base), mat)
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
            x = getcoordinates(cell); u_e = u_sol[shelldofs(cell)]
            for qp in 1:getnquadpoints(scv)
                xp  = Ferrite.spatial_coordinate(scv, qp, x)
                w_h = Ferrite.function_value(scv, qp, u_e)[3]
                err_sq += (w_h - W * sin(π*xp[1]) * sin(π*xp[2]))^2 * scv.detJdV[qp]
            end
        end
        sqrt(err_sq) / W        # relative: 0.5 is the norm of the exact field, i.e. w_h ≈ 0
    end

    # MITC6-a (T6, n=2,4,8): thickness-independent convergence. The rate is below the optimal
    # 3 of a quadratic element because the assumed shear field carries an O(h²) consistency
    # error that dominates once the bending error drops below it — the same at both
    # thicknesses, which is the point: no locking. Plain T6 is locked at coarse meshes.
    let ip6 = Lagrange{RefTriangle,2}(), qr6 = QuadratureRule{RefTriangle}(4)
        for t in [0.01, 0.001]
            errs = [ss_plate_l2err_tri(n, t, QuadraticTriangle, ip6, qr6; mitc_type=MITC6a) for n in [2, 4, 8]]
            @test all(r -> r >= 1.0, [log2(errs[i] / errs[i+1]) for i in 1:2])
            @test errs[end] < 0.02
            @test errs[2] < ss_plate_l2err_tri(4, t, QuadraticTriangle, ip6, qr6) / 5
        end
    end

    # MITC3 (T3, n=4,8,16): the displacement-based T3 is fully locked (rate ≈ 0, error ≈ 0.5,
    # i.e. w_h ≈ 0) at both thicknesses. MITC3 converges at t/L=10⁻²; at 10⁻³ Lee & Bathe
    # report residual locking, so there only the margin over the locked element is asserted.
    let ip3 = Lagrange{RefTriangle,1}(), qr3 = QuadratureRule{RefTriangle}(2)
        errs = [ss_plate_l2err_tri(n, 0.01, Triangle, ip3, qr3; mitc_type=MITC3) for n in [4, 8, 16]]
        @test all(r -> r >= 1.4, [log2(errs[i] / errs[i+1]) for i in 1:2])
        @test errs[end] < 0.03
        @test errs[end] < ss_plate_l2err_tri(16, 0.01, Triangle, ip3, qr3) / 5
        err_mitc   = ss_plate_l2err_tri(32, 0.001, Triangle, ip3, qr3; mitc_type=MITC3)
        err_nomitc = ss_plate_l2err_tri(32, 0.001, Triangle, ip3, qr3)
        @test err_nomitc > 0.3          # locked: the displacement-based T3 returns w ≈ 0
        @test err_mitc < err_nomitc / 5
    end
end
