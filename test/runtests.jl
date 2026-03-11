using FerriteShells
using LinearAlgebra
using Test
using Random

function make_scv(; qr_order=1)
    ip = Lagrange{RefQuadrilateral, 1}()
    qr = QuadratureRule{RefQuadrilateral}(qr_order)
    return ShellCellValues(qr, ip, ip)
end

# Unit square in the XY plane
const X_UNIT_SQUARE = [
    Vec{3}((0.0, 0.0, 0.0)),
    Vec{3}((1.0, 0.0, 0.0)),
    Vec{3}((1.0, 1.0, 0.0)),
    Vec{3}((0.0, 1.0, 0.0)),
]

# Helper: compute residual into a fresh zeroed vector
function residual(scv, u_vec, mat)
    re = zeros(length(u_vec))
    membrane_residuals_KL!(re, scv, u_vec, mat)
    return re
end

# Helper: compute tangent into a fresh zeroed matrix
function tangent(scv, u_vec, mat)
    ke = zeros(length(u_vec), length(u_vec))
    membrane_tangent_KL!(ke, scv, u_vec, mat)
    return ke
end

# Central-difference numerical tangent: O(ε²) accuracy
function numerical_tangent(scv, u_vec, mat; ε=1e-5)
    n = length(u_vec)
    Kfd = zeros(n, n)
    for j in 1:n
        up = copy(u_vec); up[j] += ε
        um = copy(u_vec); um[j] -= ε
        Kfd[:, j] = (residual(scv, up, mat) .- residual(scv, um, mat)) ./ (2ε)
    end
    return Kfd
end

# Rotation matrix (in-plane, about z): apply as `R(θ) ⋅ Vec{3}`
@inline R(θ=-π/2) = Tensor{2,3}([cos(θ) -sin(θ) 0; sin(θ) cos(θ) 0; 0 0 1])

# Strain energy of a single element: W = ∫ 0.5 N^{αβ} E_{αβ} dA
function element_strain_energy(scv, u_vec, mat)
    W = 0.0
    for qp in 1:getnquadpoints(scv)
        a₁, a₂, A_metric, a_metric = FerriteShells.kinematics(scv, qp, u_vec)
        E = 0.5 * (a_metric - A_metric)
        N = FerriteShells.contravariant_elasticity(mat, A_metric) ⊡ E
        W += 0.5 * (N ⊡ E) * scv.detJdV[qp]
    end
    return W
end

# Structured n×n quad mesh on the unit square [0,1]²
function unit_square_mesh(n)
    nodes = [Vec{3}((i/n, j/n, 0.0)) for j in 0:n for i in 0:n]
    cells = [Quadrilateral((j*(n+1)+i+1, j*(n+1)+i+2, (j+1)*(n+1)+i+2, (j+1)*(n+1)+i+1))
             for j in 0:n-1 for i in 0:n-1]
    return Grid(cells, Node.(nodes))
end

# Assemble external body force: r_ext += ∫ N_I f(x) dA (force per unit area)
function assemble_body_force!(r_ext, dh, scv, f_body)
    n_nodes = getnbasefunctions(scv.ip_shape)
    re = zeros(ndofs_per_cell(dh))
    for cell in CellIterator(dh)
        fill!(re, 0.0)
        reinit!(scv, cell)
        coords = getcoordinates(cell)
        for qp in 1:getnquadpoints(scv)
            ξ  = scv.qr.points[qp]
            dΩ = scv.detJdV[qp]
            x_qp = sum(Ferrite.reference_shape_value(scv.ip_shape, ξ, I) * coords[I]
                       for I in 1:n_nodes)
            f = f_body(x_qp)
            for I in 1:n_nodes
                NI = Ferrite.reference_shape_value(scv.ip_shape, ξ, I)
                re[3I-2] += NI * f[1] * dΩ
                re[3I-1] += NI * f[2] * dΩ
            end
        end
        r_ext[celldofs(cell)] .+= re
    end
end

# L2 norm of (u_h - u_exact) over the full mesh
function l2_error(dh, scv, u_h, u_exact)
    n_nodes = getnbasefunctions(scv.ip_shape)
    err_sq  = 0.0
    for cell in CellIterator(dh)
        reinit!(scv, cell)
        coords = getcoordinates(cell)
        u_e    = u_h[celldofs(cell)]
        for qp in 1:getnquadpoints(scv)
            ξ  = scv.qr.points[qp]
            dΩ = scv.detJdV[qp]
            u_h_qp = sum(Ferrite.reference_shape_value(scv.ip_shape, ξ, I) * Vec{3}((u_e[3I-2], u_e[3I-1], u_e[3I]))
                         for I in 1:n_nodes)
            x_qp   = sum(Ferrite.reference_shape_value(scv.ip_shape, ξ, I) * coords[I]
                         for I in 1:n_nodes)
            diff   = u_h_qp - u_exact(x_qp)
            err_sq += dot(diff, diff) * dΩ
        end
    end
    return sqrt(err_sq)
end

@testset "assembly.jl" begin
    @testset "membrane residuals and tangent" begin
        # make a material and a cell
        mat = LinearElastic(1.0, 0.3, 0.1)
        scv = make_scv(qr_order=1)
        reinit!(scv, X_UNIT_SQUARE)
        n_dof = 12  # 4 nodes × 3 DOFs

        # test zero displacement: should give zero residual
        re = residual(scv, zeros(n_dof), mat)
        @test norm(re) ≤ 1e-14

        # Uniform shift in x-direction — pure translation, zero strain
        u_vec = repeat([0.5, 0.0, 0.0], 4)
        re = residual(scv, u_vec, mat)
        @test norm(re) ≤ 1e-13

        # rigid body rotation should give zero membrane residual
        u = vcat([(R(-π/2)⋅xᵢ)-xᵢ for xᵢ in X_UNIT_SQUARE]...) # passed
        membrane_residuals_KL!(re, scv, u, mat)
        @test norm(re) ≤ 10eps(Float64) && sum(re) ≤ 10eps(Float64)

        # Tangent symmetry (zero displacement) should be exactly symmetric since geometric stiffness is zero.
        ke = tangent(scv, zeros(n_dof), mat)
        @test norm(ke .- ke') ≤ 1e-14 * norm(ke)

        # tangent symmetry (nonzero displacement)
        Random.seed!(1)
        u_vec = 0.05 * randn(n_dof)
        ke = tangent(scv, u_vec, mat)
        @test norm(ke .- ke') / norm(ke) ≤ 1e-12

        # FD consistency (small displacement)
        # Small displacement: geometric stiffness is negligible; material part dominates.
        Random.seed!(2)
        u_vec = 0.01 * randn(n_dof)
        ke    = tangent(scv, u_vec, mat)
        ke_fd = numerical_tangent(scv, u_vec, mat)
        rel_err = norm(ke .- ke_fd) / (norm(ke_fd) + 1e-14)
        @test rel_err < 1e-7

        # FD consistency (moderate displacement)
        # Moderate displacement: geometric stiffness is non-negligible.
        Random.seed!(3)
        u_vec = 0.3 * randn(n_dof)
        ke = tangent(scv, u_vec, mat)
        ke_fd = numerical_tangent(scv, u_vec, mat)
        rel_err = norm(ke .- ke_fd) / (norm(ke_fd) + 1e-14)
        @test rel_err < 1e-7

        # autodiff consistency
        # using ForwardDiff
        # ke_ad = ForwardDiff.jacobian(u -> residual(scv, u, mat), u_vec)
        # rel_err = norm(ke .- ke_ad) / (norm(ke_ad) + 1e-14)
        # @test rel_err < 1e-7

        # tangent positive semi-definiteness: should have no significantly negative eigenvalues.
        # For a free element, rigid-body modes yield zero eigenvalues.
        Random.seed!(4)
        u_vec = 0.01 * randn(n_dof)
        ke = tangent(scv, u_vec, mat)
        λ  = eigvals(Symmetric(ke))
        @test minimum(λ) ≥ -1e-10 * maximum(abs, λ)

        # check that higher-order rules work correctly.
        scv2 = make_scv(qr_order=2)
        reinit!(scv2, X_UNIT_SQUARE)
        re2 = residual(scv2, zeros(n_dof), mat)
        @test norm(re2) ≤ 1e-14

        # Full 2×2 Gauss integration — required to suppress hourglass modes.
        # With 1-point (reduced) integration, Q4 gains 3 spurious hourglass modes,
        # increasing the zero-eigenvalue count from 7 to 10.
        scv2 = make_scv(qr_order=2)
        reinit!(scv2, X_UNIT_SQUARE)
        ke = tangent(scv2, zeros(n_dof), mat)
        λs  = eigvals(Symmetric(ke))
        tol = 1e-10 * maximum(abs, λs)
        n_zero     = count(λ ->  abs(λ) ≤ tol, λs)
        n_positive = count(λ ->      λ  > tol, λs)
        # 4-node Q4 membrane, 4 nodes × 3 DOFs = 12 DOFs:
        #   4 zero eigenvalues — z-DOFs carry no membrane stiffness
        #   3 zero eigenvalues — in-plane rigid body modes: Tx, Ty, Rz
        #   5 positive eigenvalues — independent in-plane deformation modes
        #
        # n_zero + n_positive == n_dof implies no negative eigenvalues.
        @test n_zero     == 7
        @test n_positive == 5
        @test n_zero + n_positive == n_dof
    end
    @testset "bending residuals and tangent" begin
        # tested in test_bending.jl
        @test true
    end
    @testset "combined membrane and bending" begin
        # TODO implement combined residuals and tangent, then add tests here
        @test true
    end
    @testset "shear residuals and tangent" begin
        # TODO implement shear residuals and tangent, then add tests here
        @test true
    end
end



@testset "Uniaxial tension: Poisson contraction" begin
    # Single Q4 element (unit square). Left edge fixed in x, bottom edge fixed
    # in y (removes in-plane RBM), right edge prescribed u_x = δ. Free DOFs are
    # u_y at the two top nodes; their analytical value is -ν*δ (Poisson).
    #
    # Node layout (X_UNIT_SQUARE):
    #   node 1 (0,0) → DOFs 1,2,3    node 2 (1,0) → DOFs 4,5,6
    #   node 3 (1,1) → DOFs 7,8,9    node 4 (0,1) → DOFs 10,11,12
    δ = 1e-3
    ν = 0.3
    mat = LinearElastic(1.0e6, ν, 0.01)
    scv = make_scv(qr_order=2)
    reinit!(scv, X_UNIT_SQUARE)
    K = tangent(scv, zeros(12), mat)

    p_dofs = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12]   # prescribed DOF indices
    p_vals = [0.0, 0.0, 0.0, δ, 0.0, 0.0, δ, 0.0, 0.0, 0.0]
    f_dofs = [8, 11]                               # free: u_y at nodes 3 and 4

    u_f = K[f_dofs, f_dofs] \ (-K[f_dofs, p_dofs] * p_vals)

    # At y=1, linearised Poisson gives u_y = -ν*δ; GL correction is O(δ²) ~ 1e-6.
    @test u_f[1] ≈ -ν*δ  rtol=1e-4   # u_y at node 3 (1,1)
    @test u_f[2] ≈ -ν*δ  rtol=1e-4   # u_y at node 4 (0,1)
end

@testset "Frame-indifference: strain energy invariant under rigid rotation" begin
    # Rotate both the element mesh and the applied displacement by θ. The strain
    # energy must be identical to the unrotated configuration.
    mat = LinearElastic(1.0e6, 0.3, 0.01)
    scv = make_scv(qr_order=2)

    ε_xx, ε_yy, γ_xy = 2e-3, 1e-3, 5e-4
    u_lin(x) = Vec{3}((ε_xx*x[1] + 0.5γ_xy*x[2], 0.5γ_xy*x[1] + ε_yy*x[2], 0.0))

    reinit!(scv, X_UNIT_SQUARE)
    u_orig = vcat(u_lin.(X_UNIT_SQUARE)...)
    W_orig = element_strain_energy(scv, u_orig, mat)

    for θ in [π/6, π/4, π/3, π/2, 2π/3, π]
        Rot   = R(θ)
        x_rot = [Rot ⋅ xi for xi in X_UNIT_SQUARE]
        u_rot = vcat([Rot ⋅ u_lin(xi) for xi in X_UNIT_SQUARE]...)
        reinit!(scv, x_rot)
        W_rot = element_strain_energy(scv, u_rot, mat)
        @test W_rot ≈ W_orig  rtol=1e-10
    end
end

# Patch mesh
#
# Classic 5-element quadrilateral patch: 4 outer ring elements + 1 central
# element. The 4 inner nodes (5–8) are intentionally placed off-grid to make
# every element non-orthogonal.
#
#   4(0,10) ─────────────── 3(10,10)
#     │     8(4,7)──7(8,7)    │
#     │  E4  │   E5  │  E3   │
#     │     5(2,2)──6(8,3)    │
#     │         E1      E2    │
#   1(0,0)  ─────────────── 2(10,0)
#
#   E1:(1,2,6,5)  E2:(2,3,7,6)  E3:(3,4,8,7)
#   E4:(4,1,5,8)  E5:(5,6,7,8)  ← central element
#
# Boundary nodes : 1,2,3,4  (outer corners, on the patch perimeter)
# Interior nodes : 5,6,7,8  (inner quad corners, shared only between elements)
#
# Patch test condition: for a linear displacement field u(x,y) the internal
# forces at the 4 interior (free) nodes must be zero.  This holds iff the
# element correctly reproduces constant strain states on arbitrary quads.

function patch_grid_2(; primitive=Quadrilateral)
    nodes = [
        Vec{3}(( 0.0,  0.0, 0.0)),
        Vec{3}((10.0,  0.0, 0.0)),
        Vec{3}((10.0, 10.0, 0.0)),
        Vec{3}(( 0.0, 10.0, 0.0)),
        Vec{3}(( 2.0,  2.0, 0.0)),
        Vec{3}(( 8.0,  3.0, 0.0)),
        Vec{3}(( 8.0,  7.0, 0.0)),
        Vec{3}(( 4.0,  7.0, 0.0)),
    ]
    if primitive == Triangle
        cells = [(1,2,5), (2,6,5), (2,3,6), (3,7,6), (3,8,7),
                 (3,4,8), (4,5,8), (4,1,5), (5,6,8), (6,7,8)]
    else
        cells = [(1,2,6,5), (2,3,7,6), (3,4,8,7), (4,1,5,8),
                 (5,6,7,8)]
    end
    return Grid([primitive(c) for c in cells], Node.(nodes))
end

# ── Assembly helpers ──────────────────────────────────────────────────────────

function assemble_residual!(r, dh, scv, u, mat)
    n  = ndofs_per_cell(dh)
    re = zeros(n)
    for cell in CellIterator(dh)
        fill!(re, 0.0)
        reinit!(scv, cell)
        x   = getcoordinates(cell)
        u_e = u[celldofs(cell)]
        membrane_residuals_KL!(re, scv, u_e, mat)
        r[celldofs(cell)] .+= re
    end
end

function assemble_tangent_and_residual!(K, r, dh, scv, u, mat)
    n         = ndofs_per_cell(dh)
    ke        = zeros(n, n)
    re        = zeros(n)
    assembler = start_assemble(K, r)
    for cell in CellIterator(dh)
        fill!(ke, 0.0); fill!(re, 0.0)
        reinit!(scv, cell)
        x   = getcoordinates(cell)
        u_e = u[celldofs(cell)]
        membrane_tangent_KL!(ke, scv, u_e, mat)
        membrane_residuals_KL!(re, scv, u_e, mat)
        assemble!(assembler, celldofs(cell), ke, re)
    end
end

# Build a global DOF vector by evaluating f at every node.
# Relies on the standard Ferrite layout for ip^3 with a single :u field:
#   celldofs = [u1x, u1y, u1z, u2x, u2y, u2z, ...]  (blocked by node)
function make_u_exact(dh, f::Function)
    u = zeros(ndofs(dh))
    for cell in CellIterator(dh)
        coords = getcoordinates(cell)
        dofs   = celldofs(cell)
        for i in eachindex(coords)
            val          = f(coords[i])
            u[dofs[3i-2]] = val[1]
            u[dofs[3i-1]] = val[2]
            u[dofs[3i  ]] = val[3]
        end
    end
    return u
end


@testset "Patch test" begin
    # initial grid
    grid0 = patch_grid_2(primitive = Quadrilateral)
    # Outer corners lie on the patch perimeter; everything else is interior.
    addnodeset!(grid0, "boundary",
        x -> isapprox(x[1],  0.0, atol=1e-10) || isapprox(x[1], 10.0, atol=1e-10) ||
                isapprox(x[2],  0.0, atol=1e-10) || isapprox(x[2], 10.0, atol=1e-10))
    addnodeset!(grid0, "interior",
        x -> !( isapprox(x[1],  0.0, atol=1e-10) || isapprox(x[1], 10.0, atol=1e-10) ||
                isapprox(x[2],  0.0, atol=1e-10) || isapprox(x[2], 10.0, atol=1e-10)))

    # test frame independence by applying a rigid rotation to the entire patch
    for θ in [0.0,0.2]
        # rotate the entire patch about the z-axis by θ, keep the initial nodesets unchanged
        grid = Grid(grid0.cells, [Node(Tensors.Vec{3}(R(θ)⋅n.x)) for n in grid0.nodes]; nodesets=grid0.nodesets)

        ip  = Lagrange{RefQuadrilateral, 1}()
        qr  = QuadratureRule{RefQuadrilateral}(2)
        scv = ShellCellValues(qr, ip, ip)

        dh = DofHandler(grid)
        add!(dh, :u, ip^3)
        close!(dh)

        mat = LinearElastic(1.0e6, 0.3, 0.01)

        # Linear displacement field representing a uniform in-plane strain state.
        # Amplitudes are small so the nonlinear (GL) correction is negligible.
        ε_xx, ε_yy, γ_xy = 1.0e-3, 2.0e-3, 5.0e-4
        u_lin(x::Vec{3}) = R(θ)⋅Vec{3}((ε_xx * x[1] + 0.5γ_xy * x[2],
                                       0.5γ_xy * x[1] + ε_yy * x[2],
                                       0.0))

        # ── Test 1: interior residual is zero under the exact linear field ─────────
        # Apply u_lin to every node (interior included) and check that the
        # assembled residual at the free (interior) DOFs is numerically zero.
        @testset "Interior residual vanishes under exact linear displacement" begin
            u_exact = make_u_exact(dh, u_lin)

            r = zeros(ndofs(dh))
            assemble_residual!(r, dh, scv, u_exact, mat)

            # Use a temporary ConstraintHandler to identify free DOFs.
            ch_tmp = ConstraintHandler(dh)
            add!(ch_tmp, Dirichlet(:u, getnodeset(grid, "boundary"), x -> zero(x), [1, 2, 3]))
            close!(ch_tmp)

            # The boundary reactions can be large; only interior entries must be zero.
            @test norm(r[ch_tmp.free_dofs]) ≤ 1e-8 * norm(r)
            @test sum(r) < 1e-12
        end

        # ── Test 2: solve with linear BCs recovers the exact interior field ────────
        # Apply u_lin as Dirichlet BCs on the 4 outer corners only, assemble the
        # linearised stiffness (tangent at u = 0), and solve for the 4 inner nodes.
        # The solution must coincide with u_lin at all interior nodes.
        @testset "Linear solve with boundary data recovers interior displacements" begin
            K = allocate_matrix(dh)
            r = zeros(ndofs(dh))
            # Tangent at u=0: zero strain → zero stress → Kgeo = 0, only Kmat.
            assemble_tangent_and_residual!(K, r, dh, scv, zeros(ndofs(dh)), mat)
            # r ≡ 0 at this point (undeformed config → zero residual).

            ch = ConstraintHandler(dh)
            add!(ch, Dirichlet(:u, getnodeset(grid, "boundary"),
                    x -> u_lin(x), [1, 2, 3]))
            # The membrane has no out-of-plane (z) stiffness.  Interior z-DOFs are
            # zero-energy modes that make K singular.  Pin them to zero to regularise.
            add!(ch, Dirichlet(:u, getnodeset(grid, "interior"), x -> 0.0, [3]))
            close!(ch)
            Ferrite.update!(ch, 0.0)

            apply!(K, r, ch)
            u_solved = K \ r

            u_exact   = make_u_exact(dh, u_lin)

            err = norm(u_solved[ch.free_dofs] .- u_exact[ch.free_dofs])
            ref = norm(u_exact[ch.free_dofs])
            @test err ≤ 1e-8 * ref
        end
    end
end


@testset "Cook's membrane test" begin
    function create_cook_grid(nx, ny; primitive=Quadrilateral)
        corners = [Tensors.Vec{2}((0.0, 0.0)),
                Tensors.Vec{2}((48.0, 44.0)),
                Tensors.Vec{2}((48.0, 60.0)),
                Tensors.Vec{2}((0.0, 44.0))]
        return generate_grid(primitive, (nx, ny), corners) |> shell_grid # embed in into a 3D space
    end

    # integrate edge traction force into f
    # DOF ordering assumed: :u field first (3 DOFs per node, interleaved u,v,w)
    function assemble_traction_force!(f, dh, facetset, traction)
        edge_local_nodes = Ferrite.reference_facets(RefQuadrilateral)
        n_dpc = ndofs_per_cell(dh)
        fe    = zeros(n_dpc)
        for fc in FacetIterator(dh, facetset)
            x  = getcoordinates(fc)
            fn = fc.current_facet_id             # local facet index: 1, 2, or 3
            ia, ib = edge_local_nodes[fn]        # local node indices on this edge
            edge_len = norm(x[ib] - x[ia])
            fill!(fe, 0.0)
            # 1-point midpoint quadrature: both edge nodes receive equal weight 0.5
            # (exact for the linear shape functions used here)
            for (node, N) in ((ia, 0.5), (ib, 0.5))
                for c in 1:3    # u, v, w components
                    fe[3(node-1)+c] += N * traction[c] * edge_len
                end
            end
            f[celldofs(fc)] .+= fe
        end
    end

    # number of cells
    grid = create_cook_grid(32, 32)

    # facesets for boundary conditions
    addfacetset!(grid, "clamped", x -> norm(x[1]) ≈ 0.0)
    addfacetset!(grid, "traction", x -> norm(x[1]) ≈ 48.0)
    addnodeset!(grid, "nodes", x -> true)

    # interpolation order
    ip = Lagrange{RefQuadrilateral,1}() #to define fields only
    qr = QuadratureRule{RefQuadrilateral}(2) # avoid zero spurious modes

    # cell (shell) values
    scv = ShellCellValues(qr, ip, ip)

    # degrees of freedom for displacements (pure membrane test)
    dh = DofHandler(grid)
    add!(dh, :u, ip^3)
    close!(dh)

    # material model
    mat = LinearElastic(1.0, 1/3)

    # boundary conditions
    dbc = ConstraintHandler(dh)
    add!(dbc, Dirichlet(:u, getfacetset(dh.grid, "clamped"), x -> zero(x), [1,2,3]))
    add!(dbc, Dirichlet(:u, getnodeset(dh.grid, "nodes"), x -> [0.0], [3]))
    close!(dbc)

    # stiffness matrix assembly (for visual inspection only, not used in a solve here)
    Ke = allocate_matrix(dh)
    f = zeros(ndofs(dh))
    assemble_tangent_and_residual!(Ke, f, dh, scv, zeros(ndofs(dh)), mat)

    # test traction force assembly
    assemble_traction_force!(f, dh, getfacetset(grid, "traction"), (0.0, 1/16, 0.0))

    # apply BCs
    apply!(Ke, f, dbc)
    ue = Ke\f

    # extract solution at point
    ph     = PointEvalHandler(grid, [Tensors.Vec{3}((48.0, 60.0, 0.0))])
    u_eval = first(evaluate_at_points(ph, dh, ue, :u))
    @test all(u_eval .- [-18.5338, 24.8366, 0.0] .< 1e-3)
end

@testset "Convergence: O(h²) for smooth manufactured solution" begin
    # Manufactured solution on [0,1]²: u_x = u_y = sin(πx)sin(πy), u_z = 0.
    # Vanishes on all four edges → homogeneous Dirichlet BCs.
    # Body force derived by differentiating the linearised stress divergence.
    E_mod, ν, t = 1.0e6, 0.3, 0.01
    mat = LinearElastic(E_mod, ν, t)
    α = E_mod * t / (1 - ν^2)        # plane-stress modulus × thickness
    G = E_mod * t / (2*(1 + ν))      # in-plane shear modulus × thickness

    u_ex(x) = Vec{3}((sin(π*x[1])*sin(π*x[2]),
                       sin(π*x[1])*sin(π*x[2]),
                       0.0))

    # f_x = f_y = π²*[(α+G)*sin(πx)sin(πy) - (αν+G)*cos(πx)cos(πy)]
    # (derived from -div σ with σ from linearised strains of u_ex)
    f_body(x) = begin
        fxy = π^2 * ((α + G)*sin(π*x[1])*sin(π*x[2])
                   - (α*ν + G)*cos(π*x[1])*cos(π*x[2]))
        Vec{3}((fxy, fxy, 0.0))
    end

    errors = Float64[]
    for n in [2, 4, 8, 16]
        grid = unit_square_mesh(n)
        addnodeset!(grid, "boundary", x -> isapprox(x[1], 0.0, atol=1e-12) || isapprox(x[1], 1.0, atol=1e-12) ||
                                           isapprox(x[2], 0.0, atol=1e-12) || isapprox(x[2], 1.0, atol=1e-12))
        addnodeset!(grid, "interior", x -> !(isapprox(x[1], 0.0, atol=1e-12) || isapprox(x[1], 1.0, atol=1e-12) ||
                                             isapprox(x[2], 0.0, atol=1e-12) || isapprox(x[2], 1.0, atol=1e-12)))

        ip  = Lagrange{RefQuadrilateral, 1}()
        qr  = QuadratureRule{RefQuadrilateral}(3)   # 3-pt rule: exact for cubics
        scv = ShellCellValues(qr, ip, ip)

        dh = DofHandler(grid)
        add!(dh, :u, ip^3)
        close!(dh)

        K = allocate_matrix(dh)
        r = zeros(ndofs(dh))
        assemble_tangent_and_residual!(K, r, dh, scv, zeros(ndofs(dh)), mat)
        # At u=0: r = R_int = 0, so the rhs is purely the body force.

        r_ext = zeros(ndofs(dh))
        assemble_body_force!(r_ext, dh, scv, f_body)

        ch = ConstraintHandler(dh)
        add!(ch, Dirichlet(:u, getnodeset(grid, "boundary"), x -> zero(x), [1, 2, 3]))
        add!(ch, Dirichlet(:u, getnodeset(grid, "interior"), x -> 0.0, 3))
        close!(ch)
        Ferrite.update!(ch, 0.0)

        apply!(K, r_ext, ch)
        u_h = K \ r_ext

        push!(errors, l2_error(dh, scv, u_h, u_ex))
    end

    # Convergence rate between successive meshes: log₂(e_{2h}/e_h) ≥ 1.8 (expect ≈ 2)
    rates = [log2(errors[i] / errors[i+1]) for i in 1:length(errors)-1]
    @test all(r -> r ≥ 1.8, rates)
end

@testset "RM Cook's membrane" begin
    # RM (Reissner-Mindlin) Cook's membrane: same geometry and loading as the KL test.
    # Cook's membrane is an in-plane dominated problem; KL and RM should give identical
    # tip displacements (bending/shear DOFs are pinned everywhere, membrane governs).
    function create_cook_grid_rm(nx, ny)
        corners = [Tensors.Vec{2}((0.0, 0.0)),
                   Tensors.Vec{2}((48.0, 44.0)),
                   Tensors.Vec{2}((48.0, 60.0)),
                   Tensors.Vec{2}((0.0, 44.0))]
        return generate_grid(Quadrilateral, (nx, ny), corners) |> shell_grid
    end

    grid_rm = create_cook_grid_rm(32, 32)
    addfacetset!(grid_rm, "clamped", x -> norm(x[1]) ≈ 0.0)
    addfacetset!(grid_rm, "traction", x -> norm(x[1]) ≈ 48.0)
    addnodeset!(grid_rm, "allnodes", x -> true)

    ip_rm  = Lagrange{RefQuadrilateral, 1}()
    qr_rm  = QuadratureRule{RefQuadrilateral}(2)
    fqr_rm = FacetQuadratureRule{RefQuadrilateral}(2)
    scv_rm = ShellCellValues(qr_rm, ip_rm, ip_rm)
    mat_rm = LinearElastic(1.0, 1/3, 1.0)

    dh_rm = DofHandler(grid_rm)
    add!(dh_rm, :u, ip_rm^3)
    add!(dh_rm, :θ, ip_rm^2)
    close!(dh_rm)

    n_el_rm = ndofs_per_cell(dh_rm)
    K_rm = allocate_matrix(dh_rm)
    f_rm = zeros(ndofs(dh_rm))
    asmb_rm = start_assemble(K_rm, zeros(ndofs(dh_rm)))
    ke_rm = zeros(n_el_rm, n_el_rm); re_rm = zeros(n_el_rm)

    for cell in CellIterator(dh_rm)
        fill!(ke_rm, 0.0); fill!(re_rm, 0.0)
        reinit!(scv_rm, cell)
        x = getcoordinates(cell); u_e = zeros(n_el_rm)
        membrane_tangent_RM!(ke_rm, scv_rm, u_e, mat_rm)
        bending_tangent_RM!(ke_rm, scv_rm, u_e, mat_rm)
        assemble!(asmb_rm, shelldofs(cell), ke_rm, re_rm)
    end

    assemble_traction!(f_rm, dh_rm, getfacetset(grid_rm, "traction"),
                       ip_rm, fqr_rm, Vec{3}((0.0, 1/16, 0.0)))

    dbc_rm = ConstraintHandler(dh_rm)
    add!(dbc_rm, Dirichlet(:u, getfacetset(grid_rm, "clamped"), x -> zeros(3), [1,2,3]))
    add!(dbc_rm, Dirichlet(:θ, getfacetset(grid_rm, "clamped"), x -> zeros(2), [1,2]))
    add!(dbc_rm, Dirichlet(:u, getnodeset(grid_rm, "allnodes"), x -> [0.0], [3]))
    add!(dbc_rm, Dirichlet(:θ, getnodeset(grid_rm, "allnodes"), x -> zeros(2), [1,2]))
    close!(dbc_rm); Ferrite.update!(dbc_rm, 0.0)
    apply!(K_rm, f_rm, dbc_rm)
    ue_rm = K_rm \ f_rm

    ph_rm    = PointEvalHandler(grid_rm, [Tensors.Vec{3}((48.0, 60.0, 0.0))])
    u_tip_rm = first(evaluate_at_points(ph_rm, dh_rm, ue_rm, :u))
    @test all(u_tip_rm .- [-18.5338, 24.8366, 0.0] .< 1e-3)
end

include("test_bending.jl")
include("test_rm.jl")
include("test_utils.jl")
include("test_plate.jl")
include("test_benchmarks.jl")
