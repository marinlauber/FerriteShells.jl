# Standalone allocation-regression check — run it directly:
#
#   julia --project test/test_allocations.jl

using FerriteShells, LinearAlgebra, Test
import FerriteShells: covariant_basis, director_field

# Measure allocations of a zero-arg thunk after warm-up (compile first, then
# measure a steady call so we time the kernel, not JIT).
function alloc_of(f)
    f(); f()
    @allocated f()
end

# Per-QP helper loops (function barrier so we measure the loop body, not the
# dynamic call boundary). director_field/covariant_basis must be allocation-free.
function loop_director(scv, u_e, n_nodes, n_qp)
    s = zero(Vec{3,Float64})
    for qp in 1:n_qp
        d, d₁, d₂ = director_field(scv, qp, u_e, n_nodes)
        s += d + d₁ + d₂
    end
    s
end
function loop_covariant(scv, u_e, n_nodes, n_qp)
    s = zero(Vec{3,Float64})
    for qp in 1:n_qp
        a₁, a₂ = covariant_basis(scv, qp, u_e, n_nodes)
        s += a₁ + a₂
    end
    s
end

# elements
elements = ((Quadrilateral, RefQuadrilateral, 1, MITC4),
            (QuadraticQuadrilateral, RefQuadrilateral, 2, MITC9))

@testset "Element kernel allocations" begin
   for elem in elements
        (Q,R,O,M) = elem # unpack
        # unit element
        grid     = shell_grid(generate_grid(Q, (2, 2)))
        ip       = Lagrange{R, O}()
        qr       = QuadratureRule{R}(O+1)
        fqr      = FacetQuadratureRule{R}(O+1)
        scv_mitc = ShellCellValues(qr, ip, ip; mitc=M)
        scv      = ShellCellValues(qr, ip, ip)
        # dofs and material
        dh = DofHandler(grid)
        add!(dh, :u, ip^3); add!(dh, :θ, ip^2); close!(dh)
        mat  = LinearElastic(1.0e6, 0.3, 0.1)
        # update to match first cell
        reinit!(scv_mitc, first(CellIterator(dh)))
        n_e     = ndofs_per_cell(dh)
        n_nodes = getnbasefunctions(scv_mitc.ip_shape)
        n_qp    = getnquadpoints(scv_mitc)
        u_e     = 0.001 .* randn(n_e)
        ke      = zeros(n_e, n_e)
        re      = zeros(n_e)
        f_ext   = zeros(ndofs(dh))
        # Per-QP geometry helpers: must allocate nothing.
        @test alloc_of(() -> loop_director(scv_mitc, u_e, n_nodes, n_qp))  == 0
        @test alloc_of(() -> loop_covariant(scv_mitc, u_e, n_nodes, n_qp)) == 0
        # Follower-pressure kernels
        @test alloc_of(() -> assemble_pressure!(re, scv_mitc, u_e, 1.0))         == 0
        @test alloc_of(() -> assemble_pressure_tangent!(ke, scv_mitc, u_e, 1.0)) == 0
        # test other external loading function
        @test alloc_of(() ->assemble_traction!(f_ext, dh, getfacetset(grid, "left"), ip, fqr, Vec{3}((0.,0.,1.)))) != 0
        # mass matrix, although it's usually used once...
        @test alloc_of(() -> mass_matrix!(ke, scv_mitc, 1.0, mat)) == 0
        # Residual/tangent kernels
        @test alloc_of(() -> membrane_residuals_RM!(re, scv_mitc, u_e, mat)) == 0
        @test alloc_of(() -> bending_residuals_RM!(re, scv_mitc, u_e, mat))  == 0
        @test alloc_of(() -> membrane_tangent_RM!(ke, scv_mitc, u_e, mat))   == 0
        @test alloc_of(() -> bending_tangent_RM!(ke, scv_mitc, u_e, mat))    == 0
        # non MITC case
        @test alloc_of(() -> bending_tangent_RM!(ke, scv, u_e, mat))    == 0
        # A full per-cell assembly sweep (all four kernels) — also allocation-free.
        full() = (fill!(ke, 0.0); fill!(re, 0.0);
                  membrane_residuals_RM!(re, scv_mitc, u_e, mat); bending_residuals_RM!(re, scv_mitc, u_e, mat);
                  membrane_tangent_RM!(ke, scv_mitc, u_e, mat);   bending_tangent_RM!(ke, scv_mitc, u_e, mat))
        @test alloc_of(full) == 0
    end
end