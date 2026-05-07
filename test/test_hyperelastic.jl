
# ── Element coordinates where A_metric = I₂×₂ ────────────────────────────────
#
# HyperelasticShell currently requires that the reference covariant basis vectors
# A₁, A₂ are orthonormal (A_metric = I₂×₂) so that tr(C) in the user's W function
# gives the correct physical first invariant I₁ = A^{αβ}C_αβ + C₃₃.
# For standard unit-square Q4/Q9 elements mapped from [-1,1]² to [0,1]², the
# Jacobian is ½ and A_metric = ¼I ≠ I.  Using ±1 nodes gives A_metric = I.
#
const X_Q4_SQ = [Vec{3}((-1.0,-1.0,0.0)), Vec{3}((1.0,-1.0,0.0)),
                 Vec{3}(( 1.0, 1.0,0.0)), Vec{3}((-1.0, 1.0,0.0))]

const X_Q9_SQ = [Vec{3}((-1.0,-1.0,0.0)), Vec{3}((1.0,-1.0,0.0)),
                 Vec{3}(( 1.0, 1.0,0.0)), Vec{3}((-1.0, 1.0,0.0)),
                 Vec{3}(( 0.0,-1.0,0.0)), Vec{3}((1.0, 0.0,0.0)),
                 Vec{3}(( 0.0, 1.0,0.0)), Vec{3}((-1.0, 0.0,0.0)),
                 Vec{3}(( 0.0, 0.0,0.0))]

# ── Material definitions ───────────────────────────────────────────────────────

const μ_HE = 80.0e3          # shear modulus [Pa]
const t_HE = 1.0e-3          # shell thickness [m]

# Neo-Hookean incompressible: W = μ/2*(I₁-3)
const mat_NH = HyperelasticShell(C -> μ_HE/2 * (tr(C) - 3), t_HE)

# Mooney-Rivlin: W = c₁*(I₁-3) + c₂*(I₂-3),  I₂ = ½((trC)²-C⊡C)
const c₁_MR = 50.0e3
const c₂_MR = 30.0e3
const mat_MR = HyperelasticShell(
    C -> c₁_MR*(tr(C)-3) + c₂_MR*((tr(C)^2 - C⊡C)/2 - 3), t_HE)

# Near-incompressible LinearElastic (E=3μ, ν→0.5)
const mat_LE_HE = LinearElastic(3*μ_HE, 0.4999, t_HE)

@testset "HyperelasticShell — Neo-Hookean, Q4 RM" begin
    scv = make_q4_scv()
    reinit!(scv, X_Q4_SQ)
    n = 20   # 4 nodes × 5 DOFs

    # ── 1. Zero energy and residual at reference ──────────────────────────────
    @test FerriteShells.rm_energy(zeros(n), scv, mat_NH) == 0.0
    @test norm(rm_residual(scv, zeros(n), mat_NH)) < 1e-12

    # ── 2. Correct initial stiffness ──────────────────────────────────────────
    A = scv.A_metric[1]   # = I₂×₂ for ±1 element
    N, C = membrane_stress_and_tangent(mat_NH, A, A)
    @test norm(Array(N)) < 1e-10                               # stress-free reference
    @test isapprox(C[1,1,1,1]/t_HE, 4μ_HE, rtol=1e-6)       # C^{1111}/t = 4μ for NH
    @test isapprox(C[1,2,1,2]/t_HE,   μ_HE, rtol=1e-6)       # C^{1212}/t = μ  for NH

    # ── 3. Small-strain match with near-incompressible LinearElastic ──────────
    ε = 1e-5
    u_m = zeros(n)
    u_m[5*2-4] = ε; u_m[5*3-4] = ε    # stretch nodes 2,3 in x (ε₁₁ ≈ ε/2)
    re_NH = zeros(n); membrane_residuals_RM!(re_NH, scv, u_m, mat_NH)
    re_LE = zeros(n); membrane_residuals_RM!(re_LE, scv, u_m, mat_LE_HE)
    @test maximum(abs, re_NH .- re_LE) / (maximum(abs, re_LE) + 1e-20) < 1e-3

    # ── 4. Explicit membrane residual ≈ FD (same energy, should be exact) ─────
    re_ex = zeros(n); membrane_residuals_RM!(re_ex, scv, u_m, mat_NH)
    re_fd = zeros(n); rm_residuals_RM_FD!(re_fd, scv, u_m, mat_NH)
    @test maximum(abs, re_ex .- re_fd) / (maximum(abs, re_fd) + 1e-20) < 1e-8

    # ── 5. Rigid-body translation: energy unchanged ───────────────────────────
    u_trans = zeros(n)
    for I in 1:4; u_trans[5I-4]=3.14; u_trans[5I-3]=2.72; u_trans[5I-2]=1.41; end
    @test FerriteShells.rm_energy(u_trans, scv, mat_NH) ≈
          FerriteShells.rm_energy(zeros(n),  scv, mat_NH) rtol=1e-10

    # ── 6. Tangent symmetry ───────────────────────────────────────────────────
    u_pert = zeros(n)
    for (I, X) in enumerate(X_Q4_SQ)
        u_pert[5I-4] = 1e-4 * sin(π/2 * X[1])
        u_pert[5I-2] = 2e-4 * cos(π/2 * X[2])
        u_pert[5I-1] = 5e-5; u_pert[5I] = -3e-5
    end
    ke = rm_tangent(scv, u_pert, mat_NH)
    @test norm(ke .- ke') / (norm(ke) + 1e-14) < 1e-10

    # ── 7. Tangent FD consistency ─────────────────────────────────────────────
    ke_fd = rm_fd_tangent(scv, u_pert, mat_NH)
    @test norm(ke .- ke_fd) / (norm(ke_fd) + 1e-14) < 1e-5
end

@testset "HyperelasticShell — Neo-Hookean, Q9 RM" begin
    scv = make_q9_scv()
    reinit!(scv, X_Q9_SQ)
    n = 45   # 9 nodes × 5 DOFs

    # ── 1. Zero energy and residual at reference ──────────────────────────────
    @test FerriteShells.rm_energy(zeros(n), scv, mat_NH) == 0.0
    @test norm(rm_residual(scv, zeros(n), mat_NH)) < 1e-12

    # ── 2. Tangent symmetry ───────────────────────────────────────────────────
    u_pert = zeros(n)
    for (I, X) in enumerate(X_Q9_SQ)
        u_pert[5I-2] = 1e-3 * sin(π/2 * X[1]) * sin(π/2 * X[2])
        u_pert[5I-1] = 5e-5 * X[1]; u_pert[5I] = 5e-5 * X[2]
    end
    ke = rm_tangent(scv, u_pert, mat_NH)
    @test norm(ke .- ke') / (norm(ke) + 1e-14) < 1e-10

    # ── 3. Tangent FD consistency ─────────────────────────────────────────────
    ke_fd = rm_fd_tangent(scv, u_pert, mat_NH)
    @test norm(ke .- ke_fd) / (norm(ke_fd) + 1e-14) < 1e-5

    # ── 4. Rigid-body rotation: energy unchanged ──────────────────────────────
    θ = 0.3
    u_rot = zeros(n)
    for (I, X) in enumerate(X_Q9_SQ)
        X_rot = R(θ) ⋅ X - X
        u_rot[5I-4] = X_rot[1]; u_rot[5I-3] = X_rot[2]; u_rot[5I-2] = X_rot[3]
    end
    @test FerriteShells.rm_energy(u_rot,   scv, mat_NH) ≈
          FerriteShells.rm_energy(zeros(n), scv, mat_NH) rtol=1e-6

    # ── 5. Bending residuals agree with FD at small rotation ──────────────────
    u_b = zeros(n); for I in 1:9; u_b[5I-1] = 1e-6; end
    rb_ex = zeros(n); bending_residuals_RM!(rb_ex, scv, u_b, mat_NH)
    rb_fd = zeros(n); rm_residuals_RM_FD!(rb_fd, scv, u_b, mat_NH)
    @test maximum(abs, rb_ex .- rb_fd) / (maximum(abs, rb_fd) + 1e-20) < 5e-6
end

@testset "HyperelasticShell — incompressibility at QP" begin
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
        c_ms = SymmetricTensor{2,2}((dot(a₁,a₁), dot(a₁,a₂), dot(a₂,a₂)))
        C33 = FerriteShells.get_C33(c_ms, 0.0, 0.0)
        C   = FerriteShells.build_C3D(c_ms, 0.0, 0.0, C33)
        # For A_metric = I, C_cart = C_nat so det(C_nat) = 1 checks det(C_cart) = 1
        @test det(C) ≈ 1.0 atol=1e-12
        @test mat_NH.W(C) > 0.0   # energy is positive for non-trivial deformation
    end
end

@testset "HyperelasticShell — Mooney-Rivlin" begin
    scv = make_q4_scv()
    reinit!(scv, X_Q4_SQ)
    n = 20

    # ── 1. Zero energy at reference ───────────────────────────────────────────
    @test FerriteShells.rm_energy(zeros(n), scv, mat_MR) == 0.0

    # ── 2. Stress-free reference and correct initial stiffness ────────────────
    A = scv.A_metric[1]
    N, C = membrane_stress_and_tangent(mat_MR, A, A)
    @test norm(Array(N)) < 1e-10
    # For MR: G_3D = 2(c₁+c₂).  Incompressible plane-stress: C^{1111}/t = 4G = 8(c₁+c₂),
    # C^{1212}/t = G = 2(c₁+c₂).  (Compare NH: G=μ, C^{1111}/t=4μ, C^{1212}/t=μ.)
    @test isapprox(C[1,1,1,1]/t_HE, 8*(c₁_MR+c₂_MR), rtol=1e-6)
    @test isapprox(C[1,2,1,2]/t_HE, 2*(c₁_MR+c₂_MR), rtol=1e-6)

    # ── 3. Tangent symmetry and FD consistency ────────────────────────────────
    u_pert = zeros(n)
    for I in 1:4; u_pert[5I-4]=1e-4; u_pert[5I-2]=5e-5; u_pert[5I-1]=3e-5; u_pert[5I]=-2e-5; end
    ke    = rm_tangent(scv, u_pert, mat_MR)
    ke_fd = rm_fd_tangent(scv, u_pert, mat_MR)
    @test norm(ke .- ke') / (norm(ke) + 1e-14) < 1e-10
    @test norm(ke .- ke_fd) / (norm(ke_fd) + 1e-14) < 1e-5

    # ── 4. MR is stiffer than NH at finite strain when c₁+c₂ > μ_NH ─────────
    u_stretch = zeros(n)
    for (I, X) in enumerate(X_Q4_SQ); u_stretch[5I-4] = 0.1*(X[1]+1)/2; end
    # Initial shear modulus: NH=μ_HE=80e3, MR=c₁+c₂=80e3 → equal initially
    # but MR has extra c₂*I₂ term that increases energy faster
    @test FerriteShells.rm_energy(u_stretch, scv, mat_MR) >
          FerriteShells.rm_energy(u_stretch, scv, mat_NH)
end

@testset "HyperelasticShell — KL bending, Q9" begin
    scv9 = make_q9_scv()
    reinit!(scv9, X_Q9_SQ)
    n = 27   # 9 nodes × 3 DOFs

    # ── 1. Zero bending energy at reference ───────────────────────────────────
    @test FerriteShells.bending_energy_KL(zeros(n), scv9, mat_NH) == 0.0

    # ── 2. Pure in-plane stretch: bending energy remains zero ─────────────────
    u_mem = zeros(n)
    for (I, X) in enumerate(X_Q9_SQ); u_mem[3I-2] = 0.05*(X[1]+1)/2; end
    @test FerriteShells.bending_energy_KL(u_mem, scv9, mat_NH) ≈ 0.0 atol=1e-12

    # ── 3. Bending residual FD consistency ────────────────────────────────────
    u_b = zeros(n)
    for (I, X) in enumerate(X_Q9_SQ); u_b[3I-2] = 1e-3 * X[1]^2; end
    rb_ex = bending_residual(scv9, u_b, mat_NH)
    rb_fd = ForwardDiff.gradient(u -> FerriteShells.bending_energy_KL(u, scv9, mat_NH), u_b)
    @test norm(rb_ex .- rb_fd) / (norm(rb_fd) + 1e-20) < 1e-8

    # ── 4. Bending tangent symmetry and FD consistency ────────────────────────
    kt_ex = bending_tangent(scv9, u_b, mat_NH)
    kt_fd = bending_fd_tangent(scv9, u_b, mat_NH)
    @test norm(kt_ex .- kt_ex') / (norm(kt_ex) + 1e-14) < 1e-10
    @test norm(kt_ex .- kt_fd) / (norm(kt_fd) + 1e-14) < 1e-5

    # ── 5. Bending energy scales linearly with μ ──────────────────────────────
    mat_2μ = HyperelasticShell(C -> μ_HE*(tr(C) - 3), t_HE)   # 2×μ
    @test FerriteShells.bending_energy_KL(u_b, scv9, mat_2μ) ≈
          2 * FerriteShells.bending_energy_KL(u_b, scv9, mat_NH) rtol=1e-8
end

@testset "HyperelasticShell — MITC9" begin
    scv_mitc = ShellCellValues(QuadratureRule{RefQuadrilateral}(3),
                               Lagrange{RefQuadrilateral,2}(), Lagrange{RefQuadrilateral,2}();
                               mitc=MITC9)
    reinit!(scv_mitc, X_Q9_SQ)
    n = 45

    # ── 1. Zero energy at reference ───────────────────────────────────────────
    @test FerriteShells.rm_energy(zeros(n), scv_mitc, mat_NH) == 0.0

    # ── 2. Tangent symmetry ───────────────────────────────────────────────────
    u_pert = zeros(n)
    for (I, X) in enumerate(X_Q9_SQ)
        u_pert[5I-2] = 1e-3 * X[1] * X[2]
        u_pert[5I-1] = 2e-4 * X[1]
    end
    ke = rm_tangent(scv_mitc, u_pert, mat_NH)
    @test norm(ke .- ke') / (norm(ke) + 1e-14) < 1e-10

    # ── 3. Tangent FD consistency ─────────────────────────────────────────────
    ke_fd = rm_fd_tangent(scv_mitc, u_pert, mat_NH)
    @test norm(ke .- ke_fd) / (norm(ke_fd) + 1e-14) < 1e-5

    # ── 4. Kirchhoff mode: w = x*y (bilinear), compatible rotations φ ≈ -∂w ──
    scv_ref = make_q9_scv(); reinit!(scv_ref, X_Q9_SQ)
    u_kl = zeros(n)
    for (I, X) in enumerate(X_Q9_SQ)
        u_kl[5I-2] = 1e-3 * X[1] * X[2]
        u_kl[5I-1] = -1e-3 * X[2]    # φ₁ ≈ -∂w/∂x
        u_kl[5I  ] = -1e-3 * X[1]    # φ₂ ≈ -∂w/∂y
    end
    W_mitc   = FerriteShells.rm_energy(u_kl, scv_mitc, mat_NH)
    W_nomitc = FerriteShells.rm_energy(u_kl, scv_ref,  mat_NH)
    @test W_mitc > 0.0
    @test W_mitc ≈ W_nomitc rtol=0.05   # should be close for smooth KL mode
end
