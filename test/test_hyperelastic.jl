
# Standard element coordinates (mapped from [-1,1]² to [0,1]²).
# The frame transformation in rm_qp_energy / bending_kl_qp_energy converts the
# covariant C tensor to physical Cartesian before calling W, so A_metric ≠ I
# is fully supported.
const X_Q4_SQ = X_Q4_UNIT
const X_Q9_SQ = X_Q9_UNIT

# Material definitions
const μ_HE = 80.0e3          # shear modulus [Pa]
const t_HE = 1.0e-3          # shell thickness [m]

# Neo-Hookean incompressible: W = μ/2*(I₁-3)
const mat_NH = Hyperelastic(C -> μ_HE/2 * (tr(C) - 3), t_HE)

# Mooney-Rivlin: W = c₁*(I₁-3) + c₂*(I₂-3),  I₂ = ½((trC)²-C⊡C)
const c₁_MR = 50.0e3
const c₂_MR = 30.0e3
const mat_MR = Hyperelastic(C -> c₁_MR*(tr(C)-3) + c₂_MR*((tr(C)^2 - C⊡C)/2 - 3), t_HE)

# Near-incompressible LinearElastic (E=3μ, ν→0.5)
const mat_LE_HE = LinearElastic(3*μ_HE, 0.4999, t_HE)

function _cook_assemble_rm!(K, f, dh, scv, mat)
    n_el = ndofs_per_cell(dh)
    fill!(K.nzval, 0); fill!(f, 0)
    ke = zeros(n_el, n_el); re = zeros(n_el)
    asm = start_assemble(K, f)
    for cell in CellIterator(dh)
        fill!(ke, 0); fill!(re, 0)
        reinit!(scv, cell)
        sd  = shelldofs(cell)
        u_e = zeros(n_el)
        membrane_residuals_RM!(re, scv, u_e, mat)
        bending_residuals_RM!(re, scv, u_e, mat)
        membrane_tangent_RM!(ke, scv, u_e, mat)
        bending_tangent_RM!(ke, scv, u_e, mat)
        assemble!(asm, sd, ke, re)
    end
end

function _cook_tip_y(grid, dh, u_sol)
    tip_id = 0; dist = Inf
    for (id, node) in enumerate(grid.nodes)
        d = norm(node.x - Vec{3}((48.0, 60.0, 0.0)))
        if d < dist; dist = d; tip_id = id; end
    end
    for cell in CellIterator(dh)
        for (I, gid) in enumerate(getnodes(cell))
            gid == tip_id && return u_sol[celldofs(cell)[3I-1]]
        end
    end
    error("tip node not found")
end

function _cook_rm_solve(mat, n_mesh)
    corners = [Vec{2}((0.,0.)), Vec{2}((48.,44.)), Vec{2}((48.,60.)), Vec{2}((0.,44.))]
    grid    = generate_grid(Quadrilateral, (n_mesh, n_mesh), corners) |> shell_grid
    addfacetset!(grid, "clamped",  x -> norm(x[1]) ≈ 0.0)
    addfacetset!(grid, "traction", x -> norm(x[1]) ≈ 48.0)
    ip  = Lagrange{RefQuadrilateral,1}()
    qr  = QuadratureRule{RefQuadrilateral}(2)
    fqr = FacetQuadratureRule{RefQuadrilateral}(2)
    scv = ShellCellValues(qr, ip, ip)
    dh  = DofHandler(grid)
    add!(dh, :u, ip^3); add!(dh, :θ, ip^2); close!(dh)
    K = allocate_matrix(dh); f = zeros(ndofs(dh))
    _cook_assemble_rm!(K, f, dh, scv, mat)
    assemble_traction!(f, dh, getfacetset(grid,"traction"), ip, fqr, Vec{3}((0., 1/16, 0.)))
    dbc = ConstraintHandler(dh)
    add!(dbc, Dirichlet(:u, getfacetset(grid,"clamped"), x -> zeros(3), [1,2,3]))
    add!(dbc, Dirichlet(:θ, getfacetset(grid,"clamped"), x -> zeros(2), [1,2]))
    close!(dbc); apply!(K, f, dbc)
    _cook_tip_y(grid, dh, K \ f)
end

@testset "Hyperelastic" begin

    scv = make_q4_scv()
    reinit!(scv, X_Q4_SQ)
    n = 20   # 4 nodes × 5 DOFs

    # zero energy and residual at reference
    @test FerriteShells.energy_RM(zeros(n), scv, mat_NH) == 0.0
    @test norm(rm_residual(scv, zeros(n), mat_NH)) < 1e-12

    # stress-free reference and correct stiffness ratios
    A  = scv.A_metric[1]
    A₁ = Vec{3}(Tuple(scv.A₁[1])); A₂ = Vec{3}(Tuple(scv.A₂[1])); G₃ = scv.G₃_elem[1]
    N, C = membrane_stress_and_tangent(mat_NH, A, A, A₁, A₂, G₃)
    @test norm(Array(N)) < 1e-10                               # stress-free reference
    @test C[1,1,1,1] > 0                                       # positive stiffness
    # Natural-frame tangent scales as A_up[α,β]²; frame-independent ratio = 4 for NH.
    @test isapprox(C[1,1,1,1] / C[1,2,1,2], 4.0, rtol=1e-6)  # C^{1111}/C^{1212} = 4μ/μ = 4

    # small-strain match with near-incompressible LinearElastic
    ε = 1e-5
    u_m = zeros(n)
    u_m[5*2-4] = ε; u_m[5*3-4] = ε    # stretch nodes 2,3 in x (ε₁₁ ≈ ε/2)
    re_NH = zeros(n); membrane_residuals_RM!(re_NH, scv, u_m, mat_NH)
    re_LE = zeros(n); membrane_residuals_RM!(re_LE, scv, u_m, mat_LE_HE)
    @test maximum(abs, re_NH .- re_LE) / (maximum(abs, re_LE) + 1e-20) < 1e-3

    # Explicit membrane residual ≈ FD (same energy, should be exact)
    re_ex = zeros(n); membrane_residuals_RM!(re_ex, scv, u_m, mat_NH)
    re_fd = zeros(n); residuals_RM_FD!(re_fd, scv, u_m, mat_NH)
    @test maximum(abs, re_ex .- re_fd) / (maximum(abs, re_fd) + 1e-20) < 1e-8

    # Rigid-body translation: energy unchanged
    u_trans = zeros(n)
    for I in 1:4; u_trans[5I-4]=3.14; u_trans[5I-3]=2.72; u_trans[5I-2]=1.41; end
    @test FerriteShells.energy_RM(u_trans, scv, mat_NH) ≈ FerriteShells.energy_RM(zeros(n), scv, mat_NH) rtol=1e-10

    # tangent symmetry
    u_pert = zeros(n)
    for (I, X) in enumerate(X_Q4_SQ)
        u_pert[5I-4] = 1e-4 * sin(π/2 * X[1])
        u_pert[5I-2] = 2e-4 * cos(π/2 * X[2])
        u_pert[5I-1] = 5e-5; u_pert[5I] = -3e-5
    end
    ke = rm_tangent(scv, u_pert, mat_NH)
    @test norm(ke .- ke') / (norm(ke) + 1e-14) < 1e-10

    # tangent FD consistency
    ke_fd = rm_fd_tangent(scv, u_pert, mat_NH)
    @test norm(ke .- ke_fd) / (norm(ke_fd) + 1e-14) < 1e-5

    scv = make_q9_scv()
    reinit!(scv, X_Q9_SQ)
    n = 45   # 9 nodes × 5 DOFs

    # zero energy and residual at reference
    @test FerriteShells.energy_RM(zeros(n), scv, mat_NH) ≈ 0.0 atol=1e-12
    @test norm(rm_residual(scv, zeros(n), mat_NH)) < 1e-10

    # tangent symmetry
    u_pert = zeros(n)
    for (I, X) in enumerate(X_Q9_SQ)
        u_pert[5I-2] = 1e-3 * sin(π/2 * X[1]) * sin(π/2 * X[2])
        u_pert[5I-1] = 5e-5 * X[1]; u_pert[5I] = 5e-5 * X[2]
    end
    ke = rm_tangent(scv, u_pert, mat_NH)
    @test norm(ke .- ke') / (norm(ke) + 1e-14) < 1e-10

    # tangent FD consistency
    ke_fd = rm_fd_tangent(scv, u_pert, mat_NH)
    @test norm(ke .- ke_fd) / (norm(ke_fd) + 1e-14) < 1e-5

    # rigid-body rotation: energy unchanged
    θ = 0.3
    u_rot = zeros(n)
    for (I, X) in enumerate(X_Q9_SQ)
        X_rot = R(θ) ⋅ X - X
        u_rot[5I-4] = X_rot[1]; u_rot[5I-3] = X_rot[2]; u_rot[5I-2] = X_rot[3]
    end
    @test FerriteShells.energy_RM(u_rot,   scv, mat_NH) ≈ FerriteShells.energy_RM(zeros(n), scv, mat_NH) rtol=1e-6

    # bending residuals agree with FD at small rotation
    u_b = zeros(n); for I in 1:9; u_b[5I-1] = 1e-6; end
    rb_ex = zeros(n); bending_residuals_RM!(rb_ex, scv, u_b, mat_NH)
    rb_fd = zeros(n); residuals_RM_FD!(rb_fd, scv, u_b, mat_NH)
    @test maximum(abs, rb_ex .- rb_fd) / (maximum(abs, rb_fd) + 1e-20) < 5e-6

    # Verify det(C_cart) = 1 at every QP under a non-trivial deformation.
    scv = make_q4_scv()
    reinit!(scv, X_Q4_SQ)
    n = 20; n_nodes = 4
    u = zeros(n)
    for (I, X) in enumerate(X_Q4_SQ)
        u[5I-4] = 0.05 * (X[1] + 1)/2   # ~5% uniaxial stretch, linear in x
    end
    for qp in 1:getnquadpoints(scv)
        a₁, a₂ = FerriteShells.covariant_basis(scv, qp, u, n_nodes)
        a_metric = SymmetricTensor{2,2}((dot(a₁,a₁), dot(a₁,a₂), dot(a₂,a₂)))
        det_A = det(scv.A_metric[qp])
        C33 = FerriteShells.get_C33(a_metric, 0.0, 0.0, det_A)
        A₁q = Vec{3}(Tuple(scv.A₁[qp])); A₂q = Vec{3}(Tuple(scv.A₂[qp]))
        G₃q = scv.G₃_elem[1]
        Jinv = inv(FerriteShells._J_ref(A₁q, A₂q, G₃q))
        C_nat  = FerriteShells.build_C3D(a_metric, 0.0, 0.0, C33)
        C_cart = FerriteShells._to_C_cart(C_nat, Jinv)
        @test det(C_cart) ≈ 1.0 atol=1e-12   # det(C_cart)=1: incompressible
        @test mat_NH.W(C_cart) > 0.0           # energy positive for non-trivial deformation
    end

    scv = make_q4_scv()
    reinit!(scv, X_Q4_SQ)
    n = 20

    # zero energy at reference
    @test FerriteShells.energy_RM(zeros(n), scv, mat_MR) == 0.0

    # stress-free reference and correct stiffness ratios
    A  = scv.A_metric[1]
    A₁ = Vec{3}(Tuple(scv.A₁[1])); A₂ = Vec{3}(Tuple(scv.A₂[1])); G₃ = scv.G₃_elem[1]
    N, C = membrane_stress_and_tangent(mat_MR, A, A, A₁, A₂, G₃)
    @test norm(Array(N)) < 1e-10
    @test C[1,1,1,1] > 0
    # Frame-independent ratio: for MR G_3D=2(c₁+c₂), ratio = 4G/G = 4 (same as NH).
    @test isapprox(C[1,1,1,1] / C[1,2,1,2], 4.0, rtol=1e-6)
    # MR has twice the shear stiffness of NH (G_MR = 2(c₁+c₂) = 2*80e3 = 2*μ_NH).
    # Build NH tangent for comparison using same element.
    scv_mr = scv; A_mr = A; A₁_mr = A₁; A₂_mr = A₂; G₃_mr = G₃
    _, C_NH = membrane_stress_and_tangent(mat_NH, A_mr, A_mr, A₁_mr, A₂_mr, G₃_mr)
    @test isapprox(C[1,2,1,2] / C_NH[1,2,1,2], 2.0, rtol=1e-6)  # G_MR = 2*G_NH

    # tangent symmetry and FD consistency
    u_pert = zeros(n)
    for I in 1:4; u_pert[5I-4]=1e-4; u_pert[5I-2]=5e-5; u_pert[5I-1]=3e-5; u_pert[5I]=-2e-5; end
    ke    = rm_tangent(scv, u_pert, mat_MR)
    ke_fd = rm_fd_tangent(scv, u_pert, mat_MR)
    @test norm(ke .- ke') / (norm(ke) + 1e-14) < 1e-10
    @test norm(ke .- ke_fd) / (norm(ke_fd) + 1e-14) < 1e-5

    # MR is stiffer than NH at finite strain when c₁+c₂ > μ_NH
    u_stretch = zeros(n)
    for (I, X) in enumerate(X_Q4_SQ); u_stretch[5I-4] = 0.1*(X[1]+1)/2; end
    # Initial shear modulus: NH=μ_HE=80e3, MR=c₁+c₂=80e3 → equal initially
    # but MR has extra c₂*I₂ term that increases energy faster
    @test FerriteShells.energy_RM(u_stretch, scv, mat_MR) > FerriteShells.energy_RM(u_stretch, scv, mat_NH)

    scv9 = make_q9_scv()
    reinit!(scv9, X_Q9_SQ)
    n = 27   # 9 nodes × 3 DOFs

    # zero bending energy at reference
    @test FerriteShells.bending_energy_KL(zeros(n), scv9, mat_NH) == 0.0

    # pure in-plane stretch: bending energy remains zero
    u_mem = zeros(n)
    for (I, X) in enumerate(X_Q9_SQ); u_mem[3I-2] = 0.05*(X[1]+1)/2; end
    @test FerriteShells.bending_energy_KL(u_mem, scv9, mat_NH) ≈ 0.0 atol=1e-12

    # bending residual FD consistency
    u_b = zeros(n)
    for (I, X) in enumerate(X_Q9_SQ); u_b[3I-2] = 1e-3 * X[1]^2; end
    rb_ex = bending_residual(scv9, u_b, mat_NH)
    rb_fd = ForwardDiff.gradient(u -> FerriteShells.bending_energy_KL(u, scv9, mat_NH), u_b)
    @test norm(rb_ex .- rb_fd) / (norm(rb_fd) + 1e-20) < 1e-8

    # bending tangent symmetry and FD consistency
    kt_ex = bending_tangent(scv9, u_b, mat_NH)
    kt_fd = bending_fd_tangent(scv9, u_b, mat_NH)
    @test norm(kt_ex .- kt_ex') / (norm(kt_ex) + 1e-14) < 1e-10
    @test norm(kt_ex .- kt_fd) / (norm(kt_fd) + 1e-14) < 1e-5

    # bending energy scales linearly with μ
    mat_2μ = Hyperelastic(C -> μ_HE*(tr(C) - 3), t_HE)   # 2×μ
    @test FerriteShells.bending_energy_KL(u_b, scv9, mat_2μ) ≈ 2FerriteShells.bending_energy_KL(u_b, scv9, mat_NH) rtol=1e-8

    # Near-incompressible LE: E=3μ, ν=0.499.  Incompressible NH: same linearised G.
    μ = 1.0; t = 1.0
    mat_LE      = LinearElastic(3μ, 0.499, t)
    mat_NH_cook = Hyperelastic(C -> μ/2 * (tr(C) - 3), t)
    n_mesh = 16

    tip_LE = _cook_rm_solve(mat_LE, n_mesh)
    tip_NH = _cook_rm_solve(mat_NH_cook, n_mesh)

    @test tip_LE > 0
    @test tip_NH > 0
    @test isapprox(tip_NH, tip_LE, rtol=0.01)
    # KL reference (32×32, E=1, ν=1/3) ≈ 24.84; E=3μ=3 scales deflection by 1/3.
    @test isapprox(tip_LE, 24.84 / (3μ), rtol=0.10)

    tip_LE_fine = _cook_rm_solve(mat_LE, 24)
    tip_NH_fine = _cook_rm_solve(mat_NH_cook, 24)
    @test isapprox(tip_NH_fine, tip_LE_fine, rtol=0.005)
    @test tip_LE_fine > tip_LE

    scv_mitc = ShellCellValues(QuadratureRule{RefQuadrilateral}(3),
                               Lagrange{RefQuadrilateral,2}(), Lagrange{RefQuadrilateral,2}();
                               mitc=MITC9)
    reinit!(scv_mitc, X_Q9_SQ)
    n = 45

    # zero energy at reference
    @test FerriteShells.energy_RM(zeros(n), scv_mitc, mat_NH) ≈ 0.0 atol=1e-12

    # tangent symmetry
    u_pert = zeros(n)
    for (I, X) in enumerate(X_Q9_SQ)
        u_pert[5I-2] = 1e-3 * X[1] * X[2]
        u_pert[5I-1] = 2e-4 * X[1]
    end
    ke = rm_tangent(scv_mitc, u_pert, mat_NH)
    @test norm(ke .- ke') / (norm(ke) + 1e-14) < 1e-10

    # tangent FD consistency
    ke_fd = rm_fd_tangent(scv_mitc, u_pert, mat_NH)
    @test norm(ke .- ke_fd) / (norm(ke_fd) + 1e-14) < 1e-5

    # Kirchhoff mode: w = x*y (bilinear), compatible rotations φ ≈ -∂w
    scv_ref = make_q9_scv(); reinit!(scv_ref, X_Q9_SQ)
    u_kl = zeros(n)
    for (I, X) in enumerate(X_Q9_SQ)
        u_kl[5I-2] = 1e-3 * X[1] * X[2]
        u_kl[5I-1] = -1e-3 * X[2]    # φ₁ ≈ -∂w/∂x
        u_kl[5I  ] = -1e-3 * X[1]    # φ₂ ≈ -∂w/∂y
    end
    W_mitc   = FerriteShells.energy_RM(u_kl, scv_mitc, mat_NH)
    W_nomitc = FerriteShells.energy_RM(u_kl, scv_ref,  mat_NH)
    @test W_mitc > 0.0
    @test W_mitc ≈ W_nomitc rtol=0.05   # should be close for smooth KL mode
end
