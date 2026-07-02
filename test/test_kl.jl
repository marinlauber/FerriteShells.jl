using FerriteShells
using LinearAlgebra
using Test
using Random

# Edge-traction assembly for Cook's membrane (linear shape functions, midpoint rule).
function _assemble_edge_traction!(f, dh, facetset, traction)
    edge_local_nodes = Ferrite.reference_facets(RefQuadrilateral)
    n_dpc = ndofs_per_cell(dh); fe = zeros(n_dpc)
    for fc in FacetIterator(dh, facetset)
        x = getcoordinates(fc); fn = fc.current_facet_id
        ia, ib = edge_local_nodes[fn]
        edge_len = norm(x[ib] - x[ia])
        fill!(fe, 0.0)
        for (node, N) in ((ia, 0.5), (ib, 0.5))
            for c in 1:3; fe[3(node-1)+c] += N * traction[c] * edge_len; end
        end
        f[celldofs(fc)] .+= fe
    end
end

@testset "KL membrane" begin
    mat = LinearElastic(1.0, 0.3, 0.1)
    scv = make_q4_scv(qr_order=1)
    reinit!(scv, X_Q4_UNIT)
    n_dof = 12

    # zero residual at reference state
    @test norm(kl_residual(scv, zeros(n_dof), mat)) ≤ 1e-14

    # rigid-body translation: zero strain
    @test norm(kl_residual(scv, repeat([0.5, 0.0, 0.0], 4), mat)) ≤ 1e-13

    # rigid-body rotation: zero membrane residual
    u_rbr = vcat([(R(-π/2)⋅xᵢ) - xᵢ for xᵢ in X_Q4_UNIT]...)
    re_rbr = zeros(n_dof); membrane_residuals_KL!(re_rbr, scv, u_rbr, mat)
    @test norm(re_rbr) ≤ 10eps(Float64)

    # tangent symmetry at zero displacement (geometric stiffness vanishes)
    ke0 = kl_tangent(scv, zeros(n_dof), mat)
    @test norm(ke0 .- ke0') ≤ 1e-14 * norm(ke0)

    # tangent symmetry at nonzero displacement
    Random.seed!(1); u_rnd = 0.05 * randn(n_dof)
    let ke = kl_tangent(scv, u_rnd, mat)
        @test norm(ke .- ke') / norm(ke) ≤ 1e-12
    end

    # FD consistency: small displacement (geometric stiffness negligible)
    Random.seed!(2); u_small = 0.01 * randn(n_dof)
    let ke = kl_tangent(scv, u_small, mat), kfd = kl_fd_tangent(scv, u_small, mat)
        @test norm(ke .- kfd) / (norm(kfd) + 1e-14) < 1e-7
    end

    # FD consistency: moderate displacement (geometric stiffness non-negligible)
    Random.seed!(3); u_mod = 0.3 * randn(n_dof)
    let ke = kl_tangent(scv, u_mod, mat), kfd = kl_fd_tangent(scv, u_mod, mat)
        @test norm(ke .- kfd) / (norm(kfd) + 1e-14) < 1e-7
    end

    # higher-order quadrature: zero residual
    let scv2 = make_q4_scv(qr_order=2); reinit!(scv2, X_Q4_UNIT)
        @test norm(kl_residual(scv2, zeros(n_dof), mat)) ≤ 1e-14
    end

    # positive semi-definiteness
    Random.seed!(4); u_psd = 0.01 * randn(n_dof)
    λ = eigvals(Symmetric(kl_tangent(scv, u_psd, mat)))
    @test minimum(λ) ≥ -1e-10 * maximum(abs, λ)

    # zero-energy mode spectrum: 7 zeros + 5 positive (Q4 with full 2×2 quadrature)
    scv2 = make_q4_scv(qr_order=2); reinit!(scv2, X_Q4_UNIT)
    λs = eigvals(Symmetric(kl_tangent(scv2, zeros(n_dof), mat)))
    tol_eig = 1e-10 * maximum(abs, λs)
    @test count(λ -> abs(λ) ≤ tol_eig, λs) == 7
    @test count(λ -> λ > tol_eig, λs)      == 5

    # Poisson contraction: uniaxial tension δ at x=1, u_y = -ν*δ at y=1
    ν_p = 0.3; δ_p = 1e-3
    mat_p = LinearElastic(1.0e6, ν_p, 0.01)
    scv_p = make_q4_scv(qr_order=2); reinit!(scv_p, X_Q4_UNIT)
    K_p   = kl_tangent(scv_p, zeros(12), mat_p)
    p_dofs = [1,2,3,4,5,6,7,9,10,12]; p_vals = [0.,0.,0.,δ_p,0.,0.,δ_p,0.,0.,0.]
    u_f = K_p[[8,11],[8,11]] \ (-K_p[[8,11], p_dofs] * p_vals)
    @test u_f[1] ≈ -ν_p * δ_p  rtol=1e-4
    @test u_f[2] ≈ -ν_p * δ_p  rtol=1e-4

    # frame-indifference: strain energy invariant under rigid rotation
    mat_fi = LinearElastic(1.0e6, 0.3, 0.01)
    scv_fi = make_q4_scv(qr_order=2)
    ε_xx, ε_yy, γ_xy = 2e-3, 1e-3, 5e-4
    u_lin_fi(x) = Vec{3}((ε_xx*x[1] + 0.5γ_xy*x[2], 0.5γ_xy*x[1] + ε_yy*x[2], 0.0))
    reinit!(scv_fi, X_Q4_UNIT)
    W_orig = element_strain_energy(scv_fi, vcat(u_lin_fi.(X_Q4_UNIT)...), mat_fi)
    for θ in [π/6, π/4, π/3, π/2, 2π/3, π]
        Rot = R(θ); x_rot = [Rot ⋅ xi for xi in X_Q4_UNIT]
        reinit!(scv_fi, x_rot)
        @test element_strain_energy(scv_fi, vcat([Rot ⋅ u_lin_fi(xi) for xi in X_Q4_UNIT]...), mat_fi) ≈ W_orig rtol=1e-10
    end
end

@testset "KL membrane patch test" begin
    grid0 = patch_grid()
    addnodeset!(grid0, "boundary",
        x -> isapprox(x[1], 0.0,atol=1e-10)||isapprox(x[1],10.0,atol=1e-10)||
             isapprox(x[2], 0.0,atol=1e-10)||isapprox(x[2],10.0,atol=1e-10))
    addnodeset!(grid0, "interior",
        x -> !(isapprox(x[1], 0.0,atol=1e-10)||isapprox(x[1],10.0,atol=1e-10)||
               isapprox(x[2], 0.0,atol=1e-10)||isapprox(x[2],10.0,atol=1e-10)))

    for θ in [0.0, 0.2]
        grid = Grid(grid0.cells, [Node(Vec{3}(Tuple(R(θ) ⋅ n.x))) for n in grid0.nodes];
                    nodesets=grid0.nodesets)
        ip  = Lagrange{RefQuadrilateral,1}()
        scv = ShellCellValues(QuadratureRule{RefQuadrilateral}(2), ip, ip)
        dh  = DofHandler(grid); add!(dh, :u, ip^3); close!(dh)
        mat = LinearElastic(1.0e6, 0.3, 0.01)
        ε_xx, ε_yy, γ_xy = 1e-3, 2e-3, 5e-4
        u_lin(x::Vec{3}) = R(θ) ⋅ Vec{3}((ε_xx*x[1]+0.5γ_xy*x[2], 0.5γ_xy*x[1]+ε_yy*x[2], 0.0))

        # interior residual vanishes under exact linear displacement
        u_ex = make_u_exact(dh, u_lin)
        r = zeros(ndofs(dh)); assemble_kl_residual!(r, dh, scv, u_ex, mat)
        ch_tmp = ConstraintHandler(dh)
        add!(ch_tmp, Dirichlet(:u, getnodeset(grid, "boundary"), x -> zero(x), [1,2,3])); close!(ch_tmp)
        @test norm(r[ch_tmp.free_dofs]) ≤ 1e-8 * norm(r)
        @test sum(r) < 1e-12

        # linear solve with boundary data recovers exact interior displacements
        K = allocate_matrix(dh); r2 = zeros(ndofs(dh))
        assemble_kl_tangent!(K, r2, dh, scv, zeros(ndofs(dh)), mat)
        ch = ConstraintHandler(dh)
        add!(ch, Dirichlet(:u, getnodeset(grid, "boundary"), x -> u_lin(x), [1,2,3]))
        add!(ch, Dirichlet(:u, getnodeset(grid, "interior"), x -> 0.0, [3]))
        close!(ch); Ferrite.update!(ch, 0.0)
        apply!(K, r2, ch); u_sol = K \ r2
        @test norm(u_sol[ch.free_dofs] .- u_ex[ch.free_dofs]) ≤ 1e-8 * norm(u_ex[ch.free_dofs])
    end

    # T3 version: same 8-node geometry (scale 10), 10 triangle cells.
    # Linear field is exactly representable by T3, so patch test passes at machine precision.
    let
        grid_t3 = patch_grid(primitive=Triangle)
        addnodeset!(grid_t3, "boundary",
            x -> isapprox(x[1],0.0,atol=1e-10)||isapprox(x[1],10.0,atol=1e-10)||
                 isapprox(x[2],0.0,atol=1e-10)||isapprox(x[2],10.0,atol=1e-10))
        addnodeset!(grid_t3, "interior",
            x -> !(isapprox(x[1],0.0,atol=1e-10)||isapprox(x[1],10.0,atol=1e-10)||
                   isapprox(x[2],0.0,atol=1e-10)||isapprox(x[2],10.0,atol=1e-10)))
        ip_t3  = Lagrange{RefTriangle,1}()
        scv_t3 = ShellCellValues(QuadratureRule{RefTriangle}(2), ip_t3, ip_t3)
        dh_t3  = DofHandler(grid_t3); add!(dh_t3, :u, ip_t3^3); close!(dh_t3)
        mat_t3 = LinearElastic(1.0e6, 0.3, 0.01)
        ε_xx_t3, ε_yy_t3, γ_xy_t3 = 1e-3, 2e-3, 5e-4
        u_lin_t3(x::Vec{3}) = Vec{3}((ε_xx_t3*x[1]+0.5γ_xy_t3*x[2], 0.5γ_xy_t3*x[1]+ε_yy_t3*x[2], 0.0))

        u_ex_t3 = make_u_exact(dh_t3, u_lin_t3)
        r_t3 = zeros(ndofs(dh_t3)); assemble_kl_residual!(r_t3, dh_t3, scv_t3, u_ex_t3, mat_t3)
        ch_tmp_t3 = ConstraintHandler(dh_t3)
        add!(ch_tmp_t3, Dirichlet(:u, getnodeset(grid_t3, "boundary"), x -> zero(x), [1,2,3]))
        close!(ch_tmp_t3)
        @test norm(r_t3[ch_tmp_t3.free_dofs]) ≤ 1e-8 * norm(r_t3)

        K_t3 = allocate_matrix(dh_t3); r2_t3 = zeros(ndofs(dh_t3))
        assemble_kl_tangent!(K_t3, r2_t3, dh_t3, scv_t3, zeros(ndofs(dh_t3)), mat_t3)
        ch_t3 = ConstraintHandler(dh_t3)
        add!(ch_t3, Dirichlet(:u, getnodeset(grid_t3, "boundary"), x -> u_lin_t3(x), [1,2,3]))
        add!(ch_t3, Dirichlet(:u, getnodeset(grid_t3, "interior"), x -> 0.0, [3]))
        close!(ch_t3); Ferrite.update!(ch_t3, 0.0)
        apply!(K_t3, r2_t3, ch_t3); u_sol_t3 = K_t3 \ r2_t3
        @test norm(u_sol_t3[ch_t3.free_dofs] .- u_ex_t3[ch_t3.free_dofs]) ≤ 1e-8 * norm(u_ex_t3[ch_t3.free_dofs])
    end
end

@testset "KL Cook's membrane" begin
    corners = [Vec{2}((0.,0.)), Vec{2}((48.,44.)), Vec{2}((48.,60.)), Vec{2}((0.,44.))]
    grid = generate_grid(Quadrilateral, (32,32), corners) |> shell_grid
    addfacetset!(grid, "clamped",  x -> norm(x[1]) ≈ 0.0)
    addfacetset!(grid, "traction", x -> norm(x[1]) ≈ 48.0)
    addnodeset!(grid, "allnodes",  x -> true)

    ip  = Lagrange{RefQuadrilateral,1}()
    scv = ShellCellValues(QuadratureRule{RefQuadrilateral}(2), ip, ip)
    dh  = DofHandler(grid); add!(dh, :u, ip^3); close!(dh)
    mat = LinearElastic(1.0, 1/3)

    K = allocate_matrix(dh); f = zeros(ndofs(dh))
    assemble_kl_tangent!(K, f, dh, scv, zeros(ndofs(dh)), mat)
    _assemble_edge_traction!(f, dh, getfacetset(grid, "traction"), (0.0, 1/16, 0.0))
    dbc = ConstraintHandler(dh)
    add!(dbc, Dirichlet(:u, getfacetset(grid, "clamped"), x -> zero(x), [1,2,3]))
    add!(dbc, Dirichlet(:u, getnodeset(grid, "allnodes"), x -> [0.0],   [3]))
    close!(dbc); apply!(K, f, dbc)
    ue = K \ f

    ph     = PointEvalHandler(grid, [Vec{3}((48.0, 60.0, 0.0))])
    u_eval = first(evaluate_at_points(ph, dh, ue, :u))
    @test all(u_eval .- [-18.5338, 24.8366, 0.0] .< 1e-3)
end

@testset "KL membrane convergence" begin
    E_mod, ν, t = 1.0e6, 0.3, 0.01
    mat = LinearElastic(E_mod, ν, t)
    α   = E_mod * t / (1 - ν^2)
    G   = E_mod * t / (2*(1 + ν))
    u_ex(x)  = Vec{3}((sin(π*x[1])*sin(π*x[2]), sin(π*x[1])*sin(π*x[2]), 0.0))
    f_body(x) = let fxy = π^2 * ((α+G)*sin(π*x[1])*sin(π*x[2]) - (α*ν+G)*cos(π*x[1])*cos(π*x[2]))
        Vec{3}((fxy, fxy, 0.0))
    end

    errors = Float64[]
    for n in [2, 4, 8, 16]
        grid = unit_square_mesh(n)
        addnodeset!(grid, "boundary",
            x -> isapprox(x[1],0.0,atol=1e-12)||isapprox(x[1],1.0,atol=1e-12)||
                 isapprox(x[2],0.0,atol=1e-12)||isapprox(x[2],1.0,atol=1e-12))
        addnodeset!(grid, "interior",
            x -> !(isapprox(x[1],0.0,atol=1e-12)||isapprox(x[1],1.0,atol=1e-12)||
                   isapprox(x[2],0.0,atol=1e-12)||isapprox(x[2],1.0,atol=1e-12)))
        ip  = Lagrange{RefQuadrilateral,1}()
        scv = ShellCellValues(QuadratureRule{RefQuadrilateral}(3), ip, ip)
        dh  = DofHandler(grid); add!(dh, :u, ip^3); close!(dh)
        K = allocate_matrix(dh); r = zeros(ndofs(dh))
        assemble_kl_tangent!(K, r, dh, scv, zeros(ndofs(dh)), mat)
        r_ext = zeros(ndofs(dh)); assemble_body_force!(r_ext, dh, scv, f_body)
        ch = ConstraintHandler(dh)
        add!(ch, Dirichlet(:u, getnodeset(grid, "boundary"), x -> zero(x), [1,2,3]))
        add!(ch, Dirichlet(:u, getnodeset(grid, "interior"), x -> 0.0, 3))
        close!(ch); Ferrite.update!(ch, 0.0); apply!(K, r_ext, ch)
        push!(errors, l2_error(dh, scv, K \ r_ext, u_ex))
    end
    rates = [log2(errors[i]/errors[i+1]) for i in 1:length(errors)-1]
    @test all(r -> r ≥ 1.8, rates)
end

@testset "KL bending" begin
    mat = LinearElastic(1.0e6, 0.3, 0.01)
    scv = make_q9_scv()
    reinit!(scv, X_Q9_UNIT)
    n_dof = 27

    # zero residual at reference state
    @test bending_residual(scv, zeros(n_dof), mat) ≈ zeros(n_dof) atol=1e-14

    # rigid-body translation: zero curvature
    @test norm(bending_residual(scv, repeat([0.1, 0.2, 0.3], 9), mat)) ≤ 1e-10

    # uniform in-plane stretch: flat surface stays flat
    u_stretch = zeros(n_dof)
    for I in 1:9; u_stretch[3I-2] = 0.01*X_Q9_UNIT[I][1]; u_stretch[3I-1] = 0.01*X_Q9_UNIT[I][2]; end
    @test norm(bending_residual(scv, u_stretch, mat)) ≤ 1e-10

    # tangent symmetry at zero
    ke0 = bending_tangent(scv, zeros(n_dof), mat)
    @test norm(ke0 .- ke0') ≤ 1e-12 * norm(ke0)

    # tangent symmetry at nonzero
    Random.seed!(42); u_rnd = 0.02 * randn(n_dof)
    let ke = bending_tangent(scv, u_rnd, mat)
        @test norm(ke .- ke') / norm(ke) ≤ 1e-10
    end

    # FD consistency: small out-of-plane displacement
    Random.seed!(7); u_small = 0.001 * randn(n_dof)
    let ke = bending_tangent(scv, u_small, mat), kfd = bending_fd_tangent(scv, u_small, mat)
        @test norm(ke .- kfd) / (norm(kfd) + 1e-14) < 1e-6
    end

    # FD consistency: moderate z-displacement (curvature-dominated)
    Random.seed!(8); u_mod = zeros(n_dof)
    for I in 1:9; u_mod[3I] = 0.05*randn(); end
    let ke = bending_tangent(scv, u_mod, mat), kfd = bending_fd_tangent(scv, u_mod, mat)
        @test norm(ke .- kfd) / (norm(kfd) + 1e-14) < 1e-6
    end

    # positive semi-definiteness at zero
    λs = eigvals(Symmetric(ke0))
    @test minimum(λs) ≥ -1e-8 * maximum(abs, λs)

    # zero-energy mode spectrum: 21 zeros + 6 positive
    tol_modes = 1e-8 * maximum(abs, λs)
    @test count(λ -> abs(λ) ≤ tol_modes, λs) == 21
    @test count(λ -> λ > tol_modes, λs)      == 6

    # parabolic z-field z = x(1-x): positive bending energy (Q9 captures κ₁₁)
    u_curve = zeros(n_dof)
    for I in 1:9; u_curve[3I] = X_Q9_UNIT[I][1] * (1.0 - X_Q9_UNIT[I][1]); end
    W_curve = FerriteShells.bending_energy_KL(u_curve, scv, mat)
    @test W_curve > 0.0

    # frame-indifference: bending energy invariant under rigid rotation
    rotZ(θ) = [cos(θ) -sin(θ) 0.; sin(θ) cos(θ) 0.; 0. 0. 1.]
    rotX(θ) = [1. 0. 0.; 0. cos(θ) -sin(θ); 0. sin(θ) cos(θ)]
    for Rot in (rotZ(π/5), rotX(π/4))
        x_rot = [Vec{3}(Tuple(Rot * collect(xi))) for xi in X_Q9_UNIT]
        scv_rot = make_q9_scv(); reinit!(scv_rot, x_rot)
        u_rot = zeros(n_dof)
        for I in 1:9; u_rot[3I-2:3I] = Rot * u_curve[3I-2:3I]; end
        @test FerriteShells.bending_energy_KL(u_rot, scv_rot, mat) ≈ W_curve rtol=1e-8
    end

    # analytical energy: W ≈ 0.5*D₁₁*(−2ε)²*Area for ε·x(1−x)
    let ε = 1e-3
        D11 = mat.E * mat.thickness^3 / (12*(1 - mat.ν^2))
        @test FerriteShells.bending_energy_KL(ε*u_curve, scv, mat) ≈ 0.5*D11*4*ε^2 rtol=1e-4
    end

    # combined membrane + bending: symmetry and FD consistency
    combined_re(u) = (re = zeros(length(u)); membrane_residuals_KL!(re, scv, u, mat); bending_residuals_KL!(re, scv, u, mat); re)
    @test norm(combined_re(zeros(n_dof))) ≤ 1e-14
    ke_comb = zeros(n_dof, n_dof)
    membrane_tangent_KL!(ke_comb, scv, u_curve, mat); bending_tangent_KL!(ke_comb, scv, u_curve, mat)
    @test norm(ke_comb .- ke_comb') / norm(ke_comb) ≤ 1e-10
    ke_comb_fd = zeros(n_dof, n_dof)
    for j in 1:n_dof
        up = copy(u_curve); up[j] += 1e-5; um = copy(u_curve); um[j] -= 1e-5
        @views ke_comb_fd[:,j] = (combined_re(up) .- combined_re(um)) ./ (2e-5)
    end
    @test norm(ke_comb .- ke_comb_fd) / (norm(ke_comb_fd) + 1e-14) < 1e-6

    # curved geometry (cylindrical arc, R=5): zero reference residual + FD consistency
    let R_cyl = 5.0
        ref_st = [(0.,0.),(1.,0.),(1.,1.),(0.,1.),(0.5,0.),(1.,0.5),(0.5,1.),(0.,0.5),(0.5,0.5)]
        X_cyl  = [Vec{3}((R_cyl*sin(s/R_cyl), t, R_cyl*(1-cos(s/R_cyl)))) for (s,t) in ref_st]
        scv_cyl = make_q9_scv(); reinit!(scv_cyl, X_cyl)
        @test norm(bending_residual(scv_cyl, zeros(n_dof), mat)) ≤ 1e-12
        Random.seed!(99); u_cyl = zeros(n_dof); u_cyl[3:3:end] .= 0.01 * randn(9)
        let ke = bending_tangent(scv_cyl, u_cyl, mat), kfd = bending_fd_tangent(scv_cyl, u_cyl, mat)
            @test norm(ke .- kfd) / (norm(kfd) + 1e-14) < 1e-6
        end
    end
end

@testset "KL bending h-convergence" begin
    # Test 1: element-by-element energy on a tiled mesh (no DofHandler).
    # w(x,y) = ε·sin(πx)sin(πy), W_an = ε²·D₁₁·π⁴/2.
    let
        mat = LinearElastic(1e6, 0.3, 0.01); ε = 0.01
        D11  = mat.E * mat.thickness^3 / (12*(1-mat.ν^2))
        W_an = ε^2 * D11 * π^4 / 2

        function q9_coords(x0, x1, y0, y1)
            xm, ym = (x0+x1)/2, (y0+y1)/2
            xs = (x0,x1,x1,x0,xm,x1,xm,x0,xm); ys = (y0,y0,y1,y1,y0,ym,y1,ym,ym)
            [Vec{3}((xs[k], ys[k], 0.0)) for k in 1:9], xs, ys
        end

        scv_h = make_q9_scv()
        errors = Float64[]
        for N in [2, 4, 8, 16]
            W_FE = 0.0
            for jj in 0:N-1, ii in 0:N-1
                x_el, xs, ys = q9_coords(ii/N, (ii+1)/N, jj/N, (jj+1)/N)
                reinit!(scv_h, x_el)
                u_el = zeros(27)
                for k in 1:9; u_el[3k] = ε*sin(π*xs[k])*sin(π*ys[k]); end
                W_FE += FerriteShells.bending_energy_KL(u_el, scv_h, mat)
            end
            push!(errors, abs(W_FE - W_an))
        end
        rates = [log2(errors[i]/errors[i+1]) for i in 1:length(errors)-1]
        @test all(r -> r ≥ 1.5, rates)
    end

    # Test 2: DofHandler-based K-quadratic form. w = sin(πx)sin(πy), W_exact = ½Dπ⁴.
    let
        E, ν, t = 1e4, 0.3, 0.01
        D       = E * t^3 / (12*(1-ν^2))
        W_exact = 0.5 * D * π^4
        mat     = LinearElastic(E, ν, t)

        function kl_bending_energy_fem(n)
            ip   = Lagrange{RefQuadrilateral,2}()
            scv  = ShellCellValues(QuadratureRule{RefQuadrilateral}(4), ip, ip)
            grid = shell_grid(generate_grid(QuadraticQuadrilateral, (n,n),
                                            Vec{2}((0.,0.)), Vec{2}((1.,1.))))
            dh   = DofHandler(grid); add!(dh, :u, ip^3); close!(dh)
            n_el = ndofs_per_cell(dh); n_base = getnbasefunctions(ip)
            K = allocate_matrix(dh); asmb = start_assemble(K, zeros(ndofs(dh)))
            ke = zeros(n_el, n_el); re = zeros(n_el)
            for cell in CellIterator(dh)
                fill!(ke, 0.0); fill!(re, 0.0); reinit!(scv, cell)
                bending_tangent_KL!(ke, scv, zeros(n_el), mat)
                assemble!(asmb, celldofs(cell), ke, re)
            end
            u_h = zeros(ndofs(dh))
            for cell in CellIterator(dh)
                x = getcoordinates(cell); cd = celldofs(cell)
                for I in 1:n_base; u_h[cd[3I]] = sin(π*x[I][1])*sin(π*x[I][2]); end
            end
            0.5 * dot(u_h, K * u_h)
        end

        ws     = [kl_bending_energy_fem(n) for n in [2, 4, 8]]
        errors = abs.(ws .- W_exact)
        rates  = [log2(errors[i]/errors[i+1]) for i in 1:length(errors)-1]
        @test all(r -> r ≥ 1.5, rates)
        @test errors[end] / W_exact < 0.05
    end
end
