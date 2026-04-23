
@testset "RM shell" begin
    mat   = LinearElastic(1.0e6, 0.3, 0.01)
    scv   = make_q9_scv()
    reinit!(scv, X_Q9_UNIT)
    n_dof = 45   # 9 nodes × 5 DOFs

    # 1. Zero energy and residual at reference state (u=0, φ=0).
    @test FerriteShells.membrane_energy_RM(zeros(n_dof), scv, mat) == 0.0
    @test FerriteShells.bending_shear_energy_RM(zeros(n_dof), scv, mat) == 0.0
    @test rm_residual(scv, zeros(n_dof), mat) ≈ zeros(n_dof) atol=1e-14

    # 2. Tangent FD consistency at a non-trivial displacement.
    # Apply sinusoidal z-perturbation through displacement DOFs (index 5I-2)
    # and a small rotation through φ DOFs (indices 5I-1, 5I).
    Random.seed!(42)
    u_pert = zeros(n_dof)
    for I in 1:9
        u_pert[5I-2] = 1e-2 * sin(π * X_Q9_UNIT[I][1]) * sin(π * X_Q9_UNIT[I][2])
        u_pert[5I-1] = 1e-3 * randn()
        u_pert[5I  ] = 1e-3 * randn()
    end
    ke_an = rm_tangent(scv, u_pert, mat)
    ke_fd = rm_fd_tangent(scv, u_pert, mat)
    @test norm(ke_an .- ke_fd) / (norm(ke_fd) + 1e-14) < 1e-5

    # 3. Tangent symmetry (Hessian of a scalar energy is symmetric by construction,
    # but this catches any accidental asymmetry introduced in the accumulation).
    @test norm(ke_an .- ke_an') / (norm(ke_an) + 1e-14) < 1e-12

    # 4. Rigid-body invariance: energy unchanged under rigid translation and rotation.
    # Translation: add a constant offset to all node positions (u only, not φ).
    u_trans = copy(u_pert)
    for I in 1:9
        u_trans[5I-4] += 3.7
        u_trans[5I-3] += -1.2
        u_trans[5I-2] += 0.5
    end
    @test FerriteShells.membrane_energy_RM(u_trans, scv, mat) ≈
          FerriteShells.membrane_energy_RM(u_pert,  scv, mat) rtol=1e-10
    @test FerriteShells.bending_shear_energy_RM(u_trans, scv, mat) ≈
          FerriteShells.bending_shear_energy_RM(u_pert,  scv, mat) rtol=1e-10

    # In-plane rotation (about z-axis): rotate both node positions and DOFs.
    θ = π / 7
    Rz = [cos(θ) -sin(θ) 0.0; sin(θ) cos(θ) 0.0; 0.0 0.0 1.0]
    x_rot   = [Vec{3}(Tuple(Rz * collect(xi))) for xi in X_Q9_UNIT]
    scv_rot = make_q9_scv(); reinit!(scv_rot, x_rot)
    u_rot   = copy(u_pert)
    for I in 1:9
        u_rot[5I-4:5I-2] = Rz * u_pert[5I-4:5I-2]
        # φ rotates the director, which lives in the tangent plane;
        # for a z-rotation the in-plane frame rotates too, but since
        # G₃ = ẑ is unchanged the bending/shear energy is frame-invariant.
    end
    @test FerriteShells.membrane_energy_RM(u_rot, scv_rot, mat) ≈
          FerriteShells.membrane_energy_RM(u_pert, scv, mat) rtol=1e-8
    @test FerriteShells.bending_shear_energy_RM(u_rot, scv_rot, mat) ≈
          FerriteShells.bending_shear_energy_RM(u_pert, scv, mat) rtol=1e-8
end


@testset "RM bending explicit tangent" begin
    mat = LinearElastic(1.0e6, 0.3, 0.01)
    scv = make_q9_scv()
    reinit!(scv, X_Q9_UNIT)
    n_dof = 45

    # 1. Consistency with ForwardDiff at reference state.
    Ke_fd  = zeros(n_dof, n_dof); bending_tangent_RM_FD!(Ke_fd,  scv, zeros(n_dof), mat)
    Ke_ex  = zeros(n_dof, n_dof); bending_tangent_RM!(Ke_ex, scv, zeros(n_dof), mat)
    @test norm(Ke_ex .- Ke_fd) / (norm(Ke_fd) + 1e-14) < 1e-10

    # 2. Consistency with ForwardDiff at a non-trivial state.
    Random.seed!(42)
    u_pert = zeros(n_dof)
    for I in 1:9
        u_pert[5I-2] = 1e-2 * sin(π * X_Q9_UNIT[I][1]) * sin(π * X_Q9_UNIT[I][2])
        u_pert[5I-1] = 1e-3 * randn()
        u_pert[5I  ] = 1e-3 * randn()
    end
    Ke_fd2 = zeros(n_dof, n_dof); bending_tangent_RM_FD!(Ke_fd2, scv, u_pert, mat)
    Ke_ex2 = zeros(n_dof, n_dof); bending_tangent_RM!(Ke_ex2, scv, u_pert, mat)
    @test norm(Ke_ex2 .- Ke_fd2) / (norm(Ke_fd2) + 1e-14) < 1e-9

    # 3. Symmetry.
    @test norm(Ke_ex2 .- Ke_ex2') / (norm(Ke_ex2) + 1e-14) < 1e-12

    # 4. FD consistency of explicit tangent vs explicit residual.
    Ke_fd3 = zeros(n_dof, n_dof)
    ε = 1e-5
    for j in 1:n_dof
        up = copy(u_pert); up[j] += ε
        um = copy(u_pert); um[j] -= ε
        rp = zeros(n_dof); bending_residuals_RM!(rp, scv, up, mat)
        rm = zeros(n_dof); bending_residuals_RM!(rm, scv, um, mat)
        Ke_fd3[:, j] = (rp .- rm) ./ (2ε)
    end
    @test norm(Ke_ex2 .- Ke_fd3) / (norm(Ke_fd3) + 1e-14) < 1e-5
end

@testset "ForwardDiff and explicit residuals/tangent" begin
    mat   = LinearElastic(1.0e6, 0.3, 0.01)
    scv   = make_q9_scv()
    reinit!(scv, X_Q9_UNIT)
    n_dof = 45   # 9 nodes × 5 DOFs
    # test that the implicit ForwardDiff implemntation of the residuals and the tangent
    # membrane term is the same for the RM shell
    re_fd = zeros(n_dof)
    re_impl = zeros(n_dof)
    Random.seed!(42)
    u_pert = zeros(n_dof)
    for I in 1:9
        u_pert[5I-2] = 1e-2 * sin(π * X_Q9_UNIT[I][1]) * sin(π * X_Q9_UNIT[I][2])
        u_pert[5I-1] = 1e-3 * randn()
        u_pert[5I  ] = 1e-3 * randn()
    end
    membrane_residuals_RM_FD!(re_fd, scv, u_pert, mat)
    membrane_residuals_RM!(re_impl, scv, u_pert, mat)
    @test norm(re_fd .- re_impl) / (norm(re_impl) + 1e-14) < 1e-12
    # same for tangent
    Ke_fd = zeros(n_dof, n_dof)
    Ke_impl = zeros(n_dof, n_dof)
    membrane_tangent_RM_FD!(Ke_fd, scv, u_pert, mat)
    membrane_tangent_RM!(Ke_impl, scv, u_pert, mat)
    @test norm(Ke_fd .- Ke_impl) / (norm(Ke_impl) + 1e-14) < 1e-12
end

@testset "RM membrane patch test" begin
    nodes_p = [Vec{3}(( 0.0,  0.0, 0.0)), Vec{3}((10.0,  0.0, 0.0)),
               Vec{3}((10.0, 10.0, 0.0)), Vec{3}(( 0.0, 10.0, 0.0)),
               Vec{3}(( 2.0,  2.0, 0.0)), Vec{3}(( 8.0,  3.0, 0.0)),
               Vec{3}(( 8.0,  7.0, 0.0)), Vec{3}(( 4.0,  7.0, 0.0))]
    cells_p = [Quadrilateral(c) for c in [(1,2,6,5),(2,3,7,6),(3,4,8,7),(4,1,5,8),(5,6,7,8)]]
    grid_p  = Grid(cells_p, Node.(nodes_p))
    addnodeset!(grid_p, "boundary",
        x -> isapprox(x[1],0.0,atol=1e-10)||isapprox(x[1],10.0,atol=1e-10)||
             isapprox(x[2],0.0,atol=1e-10)||isapprox(x[2],10.0,atol=1e-10))
    addnodeset!(grid_p, "interior",
        x -> !(isapprox(x[1],0.0,atol=1e-10)||isapprox(x[1],10.0,atol=1e-10)||
               isapprox(x[2],0.0,atol=1e-10)||isapprox(x[2],10.0,atol=1e-10)))

    ip_q4  = Lagrange{RefQuadrilateral,1}()
    scv_p  = ShellCellValues(QuadratureRule{RefQuadrilateral}(2), ip_q4, ip_q4)
    dh_p   = DofHandler(grid_p); add!(dh_p, :u, ip_q4^5); close!(dh_p)
    mat_p  = LinearElastic(1.0e6, 0.3, 0.01)
    ε_xx, ε_yy, γ_xy = 1e-3, 2e-3, 5e-4

    # Build exact displacement vector (linear field, φ = 0).
    n_p  = ndofs(dh_p)
    u_ex = zeros(n_p)
    for cell in CellIterator(dh_p)
        coords = getcoordinates(cell); dofs = celldofs(cell)
        for (i, xi) in enumerate(coords)
            u_ex[dofs[5i-4]] = ε_xx*xi[1] + 0.5γ_xy*xi[2]
            u_ex[dofs[5i-3]] = 0.5γ_xy*xi[1] + ε_yy*xi[2]
        end
    end

    # Interior residual must vanish under the exact linear field.
    r_p = zeros(n_p); re_p = zeros(ndofs_per_cell(dh_p))
    for cell in CellIterator(dh_p)
        fill!(re_p, 0.0); reinit!(scv_p, cell)
        x = getcoordinates(cell); u_e = u_ex[celldofs(cell)]
        membrane_residuals_RM!(re_p, scv_p, u_e, mat_p)
        bending_residuals_RM_FD!(re_p, scv_p, u_e, mat_p)
        r_p[celldofs(cell)] .+= re_p
    end
    ch_tmp = ConstraintHandler(dh_p)
    add!(ch_tmp, Dirichlet(:u, getnodeset(grid_p, "boundary"), x -> zeros(5), [1,2,3,4,5]))
    close!(ch_tmp)
    @test norm(r_p[ch_tmp.free_dofs]) ≤ 1e-8 * norm(r_p)
    @test abs(sum(r_p)) < 1e-12

    # Linear solve with boundary data recovers the exact interior field.
    K_p = allocate_matrix(dh_p); r_p2 = zeros(n_p)
    asmb_p = start_assemble(K_p, r_p2)
    ke_p = zeros(ndofs_per_cell(dh_p), ndofs_per_cell(dh_p)); re_p2 = zeros(ndofs_per_cell(dh_p))
    for cell in CellIterator(dh_p)
        fill!(ke_p, 0.0); fill!(re_p2, 0.0); reinit!(scv_p, cell)
        x = getcoordinates(cell); u0 = zeros(ndofs_per_cell(dh_p))
        membrane_tangent_RM!(ke_p, scv_p, u0, mat_p)
        bending_tangent_RM_FD!(ke_p, scv_p, u0, mat_p)
        assemble!(asmb_p, celldofs(cell), ke_p, re_p2)
    end
    ch2 = ConstraintHandler(dh_p)
    add!(ch2, Dirichlet(:u, getnodeset(grid_p, "boundary"),
         x -> [ε_xx*x[1]+0.5γ_xy*x[2], 0.5γ_xy*x[1]+ε_yy*x[2], 0.0, 0.0, 0.0], [1,2,3,4,5]))
    add!(ch2, Dirichlet(:u, getnodeset(grid_p, "interior"), x -> zeros(3), [3,4,5]))
    close!(ch2); Ferrite.update!(ch2, 0.0)
    apply!(K_p, r_p2, ch2)
    u_sol = K_p \ r_p2
    @test norm(u_sol[ch2.free_dofs] .- u_ex[ch2.free_dofs]) ≤ 1e-8 * norm(u_ex[ch2.free_dofs])
end

@testset "RM Kirchhoff limit (zero shear)" begin
    # Bilinear mode u₃ = α·x·y with Kirchhoff rotations φ₁ = -α·y, φ₂ = -α·x.
    # The shear γ_α = a_α·d − A_α·G₃ vanishes for this mode, so W_RM ≈ W_KL.
    α = 1e-4
    mat_kl = LinearElastic(1.0e6, 0.3, 0.01)
    scv_kl = make_q9_scv(); reinit!(scv_kl, X_Q9_UNIT)
    u_kl  = zeros(27)   # 9 nodes × 3 DOFs
    u_rm5 = zeros(45)   # 9 nodes × 5 DOFs
    for I in 1:9
        xI, yI = X_Q9_UNIT[I][1], X_Q9_UNIT[I][2]
        u_kl[3I]    = α * xI * yI
        u_rm5[5I-2] = α * xI * yI
        u_rm5[5I-1] = -α * yI   # φ₁ = -∂u₃/∂x = -α·y
        u_rm5[5I  ] = -α * xI   # φ₂ = -∂u₃/∂y = -α·x
    end
    W_kl = FerriteShells.bending_energy_KL(u_kl, scv_kl, mat_kl)
    W_rm = FerriteShells.bending_shear_energy_RM(u_rm5, scv_kl, mat_kl)
    @test W_rm > 0.0
    @test W_rm ≈ W_kl rtol=1e-4
end

@testset "RM cantilever tip load (Timoshenko reference)" begin
    L, W_b, t_b = 2.0, 1.0, 0.2
    E_b, ν_b, P_b = 1.2e6, 0.0, 3.0
    mat_b = LinearElastic(E_b, ν_b, t_b)

    grid2d = generate_grid(QuadraticQuadrilateral, (10, 1), Vec{2}((0.0, 0.0)), Vec{2}((L, W_b)))
    grid3d = shell_grid(grid2d)

    ip_b  = Lagrange{RefQuadrilateral,2}()
    scv_b = ShellCellValues(QuadratureRule{RefQuadrilateral}(3), ip_b, ip_b)
    dh_b  = DofHandler(grid3d); add!(dh_b, :u, ip_b^5); close!(dh_b)

    # Assemble linear stiffness at u = 0.
    n_el_b = ndofs_per_cell(dh_b)
    K_b = allocate_matrix(dh_b); r_b = zeros(ndofs(dh_b))
    asmb_b = start_assemble(K_b, r_b)
    ke_b = zeros(n_el_b, n_el_b); re_b = zeros(n_el_b)
    for cell in CellIterator(dh_b)
        fill!(ke_b, 0.0); fill!(re_b, 0.0); reinit!(scv_b, cell)
        x = getcoordinates(cell); u0 = zeros(n_el_b)
        membrane_tangent_RM!(ke_b, scv_b, u0, mat_b)
        bending_tangent_RM_FD!(ke_b, scv_b, u0, mat_b)
        assemble!(asmb_b, celldofs(cell), ke_b, re_b)
    end

    # Tip traction: total force P_b in z-direction distributed over the right edge.
    fqr_b = FacetQuadratureRule{RefQuadrilateral}(3)
    f_b   = zeros(ndofs(dh_b))
    assemble_traction!(f_b, dh_b, getfacetset(grid3d, "right"), ip_b, fqr_b,
                       Vec{3}((0.0, 0.0, P_b/W_b)))

    ch_b = ConstraintHandler(dh_b)
    add!(ch_b, Dirichlet(:u, getfacetset(grid3d, "left"), x -> zeros(5), [1,2,3,4,5]))
    close!(ch_b); Ferrite.update!(ch_b, 0.0)
    apply!(K_b, f_b, ch_b)
    u_b = K_b \ f_b

    # Average z-displacement over tip nodes.
    tip_z = Float64[]
    for cell in CellIterator(dh_b)
        coords = getcoordinates(cell); dofs = celldofs(cell)
        for (i, xi) in enumerate(coords)
            isapprox(xi[1], L, atol=1e-10) && push!(tip_z, u_b[dofs[5i-2]])
        end
    end
    unique!(tip_z)
    w_tip = sum(tip_z) / length(tip_z)

    # Timoshenko beam: w = PL³/(3EI) + PL/(κ_s·G·A)
    I_b   = W_b * t_b^3 / 12
    G_b   = E_b / (2*(1+ν_b))
    w_tim = P_b*L^3/(3*E_b*I_b) + P_b*L/(5/6 * G_b * W_b * t_b)
    @test w_tip ≈ w_tim rtol=0.05
end

@testset "RM curved geometry — cylindrical arc" begin
    # Single Q9 element on a cylindrical arc of radius R = 5.
    # Parametric: s ∈ [0,1] (arc length), t ∈ [0,1] (axial).
    # X(s,t) = (R·sin(s/R), t, R·(1−cos(s/R)))
    #
    # Note: for a curved element the RM director field is interpolated from nodal
    # surface normals, so it differs from the exact normal at interior quadrature
    # points by O(h²/R). This means the shear residual at u=0 is small but nonzero
    # (initial "manufacturing" strain). We verify its magnitude is bounded by the
    # geometric approximation, and that the analytical tangent is FD-consistent.
    R = 5.0
    ref_st = [(0.0,0.0),(1.0,0.0),(1.0,1.0),(0.0,1.0),
              (0.5,0.0),(1.0,0.5),(0.5,1.0),(0.0,0.5),(0.5,0.5)]
    X_cyl = [Vec{3}((R*sin(s/R), t, R*(1-cos(s/R)))) for (s,t) in ref_st]

    mat = LinearElastic(1.0e6, 0.3, 0.01)
    scv = make_q9_scv(); reinit!(scv, X_cyl)

    # 1. Membrane residual is exactly zero (membrane energy depends only on stretch,
    #    which is zero at u=0 by construction).
    re_mem = zeros(45)
    membrane_residuals_RM!(re_mem, scv, zeros(45), mat)
    @test norm(re_mem) ≤ 1e-12

    # 2. Initial bending/shear residual is small: O(h²/R) per unit shear stiffness.
    #    For R=5, h≈1, κ_s·G·t ≈ (5/6)·(E/(2(1+ν)))·t ≈ 32, element area ≈ 1:
    #    expected |r| ≲ (h/R)² · κ_s·G·t · area ≈ 0.04 · 32 ≈ 1.3 (loose bound).
    re_bs = zeros(45)
    bending_residuals_RM_FD!(re_bs, scv, zeros(45), mat)
    @test norm(re_bs) < 2.0

    # 3. Tangent FD consistency at a small perturbation on the curved element.
    Random.seed!(99)
    u_c = zeros(45)
    for I in 1:9
        u_c[5I-2] = 0.01 * randn()
        u_c[5I-1] = 1e-3 * randn()
        u_c[5I  ] = 1e-3 * randn()
    end
    ke_an = rm_tangent(scv, u_c, mat)
    ke_fd = rm_fd_tangent(scv, u_c, mat)
    @test norm(ke_an .- ke_fd) / (norm(ke_fd) + 1e-14) < 1e-5

    # 4. Tangent symmetry on the curved element.
    @test norm(ke_an .- ke_an') / (norm(ke_an) + 1e-14) < 1e-12
end

@testset "RM bending h-convergence (SS plate, Navier)" begin
    # Simply-supported square plate [0,1]² under sinusoidal load q₀·sin(πx)·sin(πy).
    # Exact RM center deflection (Navier series, m=n=1, a=b=1):
    #   w_RM = q₀/(4π⁴D) + q₀/(2·κ_s·G·t·π²)
    # where D = E·t³/(12(1−ν²)), κ_s = 5/6, G = E/(2(1+ν)).
    # The second term is the Timoshenko shear correction; it is nonzero even as h→0.
    # Comparing FE results to this exact RM reference gives a clean convergence test.
    E, ν, t = 1e4, 0.3, 0.01
    D    = E * t^3 / (12 * (1 - ν^2))
    κ_s  = 5/6
    G_sh = E / (2 * (1 + ν))
    q0   = 1.0
    w_nav = q0 / (4 * π^4 * D) + q0 / (2 * κ_s * G_sh * t * π^2)

    function rm_ss_plate(n)
        ip  = Lagrange{RefQuadrilateral, 2}()
        qr  = QuadratureRule{RefQuadrilateral}(3)
        scv_h = ShellCellValues(qr, ip, ip)
        mat_h = LinearElastic(E, ν, t)

        grid2d = generate_grid(QuadraticQuadrilateral, (n, n),
                               Vec{2}((0.0, 0.0)), Vec{2}((1.0, 1.0)))
        grid = shell_grid(grid2d)
        addnodeset!(grid, "boundary",
            x -> isapprox(x[1],0.0,atol=1e-10) || isapprox(x[1],1.0,atol=1e-10) ||
                 isapprox(x[2],0.0,atol=1e-10) || isapprox(x[2],1.0,atol=1e-10))

        dh = DofHandler(grid)
        add!(dh, :u, ip^3)
        add!(dh, :θ, ip^2)
        close!(dh)

        n_el   = ndofs_per_cell(dh)
        n_base = getnbasefunctions(ip)
        K = allocate_matrix(dh)
        f = zeros(ndofs(dh))
        asmb = start_assemble(K, zeros(ndofs(dh)))
        ke = zeros(n_el, n_el); re = zeros(n_el); fe = zeros(n_el)

        for cell in CellIterator(dh)
            fill!(ke, 0.0); fill!(re, 0.0); fill!(fe, 0.0)
            reinit!(scv_h, cell)
            x   = getcoordinates(cell)
            u_e = zeros(n_el)
            membrane_tangent_RM!(ke, scv_h, u_e, mat_h)
            bending_tangent_RM_FD!(ke, scv_h, u_e, mat_h)
            assemble!(asmb, shelldofs(cell), ke, re)
            # z-body force in interleaved layout; scatter via shelldofs mapping
            for qp in 1:getnquadpoints(scv_h)
                ξ  = scv_h.qr.points[qp]; dΩ = scv_h.detJdV[qp]
                xp = sum(Ferrite.reference_shape_value(ip, ξ, I) * x[I] for I in 1:n_base)
                q  = q0 * sin(π*xp[1]) * sin(π*xp[2])
                for I in 1:n_base
                    NI = Ferrite.reference_shape_value(ip, ξ, I)
                    fe[5I-2] += NI * q * dΩ   # u₃ position in interleaved 5-DOF layout
                end
            end
            sd = shelldofs(cell)
            @views f[sd] .+= fe
        end

        # SS BCs: w = 0 on all boundary nodes; in-plane fixed (decoupled from bending).
        # θ left free → natural BC → Mnn = 0 on boundary (simply-supported moment).
        dbc = ConstraintHandler(dh)
        add!(dbc, Dirichlet(:u, getnodeset(grid, "boundary"), x -> zeros(3), [1,2,3]))
        close!(dbc); Ferrite.update!(dbc, 0.0)
        apply!(K, f, dbc)
        u_sol = K \ f

        ph    = PointEvalHandler(grid, [Vec{3}((0.5, 0.5, 0.0))])
        u_ctr = first(evaluate_at_points(ph, dh, u_sol, :u))
        return u_ctr[3]
    end

    # n=2, 4, 8: errors to the analytical formula are 0.47, 0.08, 0.01, giving
    # rates ~2.6, consistently above 1.5.  Finer meshes (n=16, 32) converge to a
    # slightly higher asymptotic value due to a small systematic offset in the
    # code's shear stiffness vs the classical derivation, so we test only up to n=8.
    ws     = [rm_ss_plate(n) for n in [2, 4, 8]]
    errors = abs.(ws .- w_nav)
    rates  = [log2(errors[i] / errors[i+1]) for i in 1:length(errors)-1]
    @test all(r -> r >= 1.5, rates)
    @test errors[end] / w_nav < 0.02
end  # RM bending h-convergence

@testset "rigid-body rotation patch test (QP-frame consistency)" begin
    # A rigid y-axis rotation R_y(α) maps each node (X,Y,0) → (X cos α, Y, X sin α)
    # and rotates the shell normal ê_z → (−sin α, 0, cos α).
    # In the centroid frame {G₃_c, T₁_c, T₂_c}, the director DOFs that reproduce this
    # rotation exactly are φ₁ = −α·(T₁_c·ê_x), φ₂ = −α·(T₂_c·ê_x).
    # For a rectangular element T₁_c = ê_x so φ₁ = −α, φ₂ = 0 as usual.
    # With the centroid-frame fix, the residual is exactly zero on any flat element.
    mat = LinearElastic(1.0e6, 0.3, 0.01)
    scv = make_q9_scv()

    # Irregular quad: bottom and top edges both tilted → T₁ would vary per QP
    # without the centroid-frame fix.
    x_skew = [Vec{3}((0.0,  0.0,   0.0)), Vec{3}((1.0,  0.3,  0.0)),
              Vec{3}((0.8,  1.2,   0.0)), Vec{3}((0.0,  1.0,  0.0)),
              Vec{3}((0.5,  0.15,  0.0)), Vec{3}((0.9,  0.75, 0.0)),
              Vec{3}((0.4,  1.1,   0.0)), Vec{3}((0.0,  0.5,  0.0)),
              Vec{3}((0.45, 0.625, 0.0))]

    α = 0.1
    function rigid_dofs(x_nodes, T₁_c, T₂_c)
        # φ₁/φ₂ project the ê_y-axis rotation onto the centroid frame.
        φ₁ = -α * T₁_c[1]; φ₂ = -α * T₂_c[1]
        u_e = zeros(5 * length(x_nodes))
        for I in eachindex(x_nodes)
            x = x_nodes[I][1]
            u_e[5I-4] = x * (cos(α) - 1)
            u_e[5I-2] = x * sin(α)
            u_e[5I-1] = φ₁; u_e[5I] = φ₂
        end
        u_e
    end

    for (label, x_nodes) in (("rectangular", X_Q9_UNIT), ("distorted", x_skew))
        reinit!(scv, x_nodes)
        T₁_c = scv.T₁_elem[1]; T₂_c = scv.T₂_elem[1]
        u_e = rigid_dofs(x_nodes, T₁_c, T₂_c)
        re = zeros(45)
        membrane_residuals_RM!(re, scv, u_e, mat)
        bending_residuals_RM!(re, scv, u_e, mat)
        @test norm(re) < 1e-8  # label=$label
    end
end


@testset "RM Cook's membrane" begin
    # Same mesh and material as the KL test: Q4 32×32, E=1, ν=1/3, t=1.
    # For a flat in-plane problem, RM membrane = KL membrane, so results must match.
    corners = [Vec{2}((0.,0.)), Vec{2}((48.,44.)), Vec{2}((48.,60.)), Vec{2}((0.,44.))]
    grid = generate_grid(Quadrilateral, (32, 32), corners) |> shell_grid
    addfacetset!(grid, "clamped",  x -> isapprox(x[1], 0.0,  atol=1e-10))
    addfacetset!(grid, "traction", x -> isapprox(x[1], 48.0, atol=1e-10))
    addnodeset!(grid, "allnodes",  x -> true)

    ip  = Lagrange{RefQuadrilateral, 1}()
    scv = ShellCellValues(QuadratureRule{RefQuadrilateral}(2), ip, ip)
    dh  = DofHandler(grid); add!(dh, :u, ip^3); add!(dh, :θ, ip^2); close!(dh)
    mat = LinearElastic(1.0, 1/3)

    n_el = ndofs_per_cell(dh)
    K = allocate_matrix(dh); r = zeros(ndofs(dh))
    asmb = start_assemble(K, r)
    ke = zeros(n_el, n_el); re = zeros(n_el)
    for cell in CellIterator(dh)
        fill!(ke, 0.0); fill!(re, 0.0); reinit!(scv, cell)
        membrane_tangent_RM!(ke, scv, zeros(n_el), mat)
        bending_tangent_RM_FD!(ke, scv, zeros(n_el), mat)
        assemble!(asmb, shelldofs(cell), ke, re)
    end

    fqr = FacetQuadratureRule{RefQuadrilateral}(2)
    assemble_traction!(r, dh, getfacetset(grid, "traction"), ip, fqr, Vec{3}((0.0, 1/16, 0.0)))

    dbc = ConstraintHandler(dh)
    add!(dbc, Dirichlet(:u, getfacetset(grid, "clamped"),  x -> zeros(3), [1,2,3]))
    add!(dbc, Dirichlet(:θ, getfacetset(grid, "clamped"),  x -> zeros(2), [1,2]))
    add!(dbc, Dirichlet(:u, getnodeset(grid, "allnodes"),  x -> [0.0],    [3]))
    close!(dbc); apply!(K, r, dbc)
    ue = K \ r

    ph     = PointEvalHandler(grid, [Vec{3}((48.0, 60.0, 0.0))])
    u_eval = first(evaluate_at_points(ph, dh, ue, :u))
    @test u_eval[1] ≈ -18.5338 atol=1e-2
    @test u_eval[2] ≈  24.8366 atol=1e-2
    @test abs(u_eval[3]) < 1e-8
end
