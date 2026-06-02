# Standalone allocation-regression check for the hot RM/MITC element kernels.
# NOT part of the test suite (not included from test/runtests.jl) — run it directly:
#
#   julia --project test/test_allocations.jl
#
# Rationale: the per-cell assembly kernels are called millions of times in the
# dynamic/inflation drivers, so heap allocation there drives GC working-set growth
# (and OOM on large meshes). These bounds catch regressions such as the
# `Vec{3,T}(Tuple(scv.G₃_elem[I]))` boxing that previously cost ~5 KB/QP in
# `director_field`, or a closure-built material tangent.
#
# Everything runs inside `run_alloc_tests()` so the measured closures capture
# *local* variables (top-level captures are type-unstable and allocate spuriously).

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
loop_director(scv, u_e, n_nodes, n_qp) =
    (s = zero(Vec{3,Float64}); for qp in 1:n_qp; d, d₁, d₂ = director_field(scv, qp, u_e, n_nodes); s += d + d₁ + d₂; end; s)
loop_covariant(scv, u_e, n_nodes, n_qp) =
    (s = zero(Vec{3,Float64}); for qp in 1:n_qp; a₁, a₂ = covariant_basis(scv, qp, u_e, n_nodes); s += a₁ + a₂; end; s)

function run_alloc_tests()
    # Q9 + MITC9 unit element (mirrors make_q9_scv from the test suite).
    grid = shell_grid(generate_grid(QuadraticQuadrilateral, (2, 2)))
    ip   = Lagrange{RefQuadrilateral, 2}()
    qr   = QuadratureRule{RefQuadrilateral}(3)
    scv  = ShellCellValues(qr, ip, ip; mitc=MITC9)
    dh   = DofHandler(grid)
    add!(dh, :u, ip^3); add!(dh, :θ, ip^2); close!(dh)
    mat  = LinearElastic(0.35e6, 0.3, 0.0002)

    reinit!(scv, first(CellIterator(dh)))
    n_e     = ndofs_per_cell(dh)
    n_nodes = getnbasefunctions(scv.ip_shape)
    n_qp    = getnquadpoints(scv)
    u_e     = 0.001 .* randn(n_e)
    ke      = zeros(n_e, n_e)
    re      = zeros(n_e)

    @testset "element kernel allocations (Q9 + MITC9, LinearElastic)" begin
        # Per-QP geometry helpers: must allocate nothing.
        @test alloc_of(() -> loop_director(scv, u_e, n_nodes, n_qp))  == 0
        @test alloc_of(() -> loop_covariant(scv, u_e, n_nodes, n_qp)) == 0

        # Follower-pressure kernels are already allocation-free.
        @test alloc_of(() -> assemble_pressure!(re, scv, u_e, 1.0))         == 0
        @test alloc_of(() -> assemble_pressure_tangent!(ke, scv, u_e, 1.0)) == 0

        # Residual/tangent kernels are fully allocation-free: geometry helpers use
        # the stored Float64 frames directly (no Tuple boxing) and the MITC tangent
        # reuses scratch buffers held in the MITC object. Were ~5.6 KB (membrane/
        # residual) and ~52–60 KB (bending) before those fixes.
        @test alloc_of(() -> membrane_residuals_RM!(re, scv, u_e, mat)) == 0
        @test alloc_of(() -> bending_residuals_RM!(re, scv, u_e, mat))  == 0
        @test alloc_of(() -> membrane_tangent_RM!(ke, scv, u_e, mat))   == 0
        @test alloc_of(() -> bending_tangent_RM!(ke, scv, u_e, mat))    == 0

        # A full per-cell assembly sweep (all four kernels) — also allocation-free.
        full() = (fill!(ke, 0.0); fill!(re, 0.0);
                  membrane_residuals_RM!(re, scv, u_e, mat); bending_residuals_RM!(re, scv, u_e, mat);
                  membrane_tangent_RM!(ke, scv, u_e, mat);   bending_tangent_RM!(ke, scv, u_e, mat))
        @test alloc_of(full) == 0
    end
end

run_alloc_tests()
