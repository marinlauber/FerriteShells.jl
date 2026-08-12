# Standalone allocation-regression check for the Hyperelastic material model —
# run it directly:
#
#   julia --project test/test_hyperelastic_allocations.jl
#
# Motivation: a Newton solve with Hyperelastic(...; incompressible=false) on a large
# mesh was reported to run out of memory ("allocation overflow"). The suspect is the
# plane-stress C₃₃ condensation (`_C33_planestress`), which runs its own Newton
# iteration *underneath* the nested ForwardDiff `gradient(∇W, c_ms, :all)` used to
# get the membrane tangent — i.e. Dual-of-Dual-of-Dual arithmetic. If that inner loop
# allocates, every quadrature point of every element of every Newton iteration of the
# outer FE solve pays for it, and garbage scales with (n_elem × n_qp × n_outer_iter).
# This script isolates the material-model layer (no mesh/assembly) so the source of
# any allocation can be pinned down.

using FerriteShells, LinearAlgebra, Test, Tensors, ForwardDiff
import FerriteShells: _C33_planestress, _J_ref, get_C33

alloc_of(f) = (f(); f(); @allocated f())

# ---------------------------------------------------------------------------
# Materials: NH is incompressible (analytic C₃₃, no Newton) — the control.
# SVK-ps is compressible and forces the Newton condensation (incompressible=false)
# — the suspect. SVK-inc is the same energy under the analytic path, isolating
# "compressible energy" from "Newton condensation" as separate variables.
# ---------------------------------------------------------------------------
const μ_HE = 80.0e3
const t_HE = 1.0e-3
const mat_NH = Hyperelastic(C -> μ_HE/2 * (tr(C) - 3), t_HE)

const E_SVK = 1.0e6
const ν_SVK = 0.3
const λ_SVK = E_SVK*ν_SVK/((1 + ν_SVK)*(1 - 2ν_SVK))
const μ_SVK = E_SVK/(2*(1 + ν_SVK))
W_SVK(C) = (Eg = (C - one(C))/2; λ_SVK/2 * tr(Eg)^2 + μ_SVK * (Eg ⊡ Eg))
const mat_SVK_ps  = Hyperelastic(W_SVK, t_HE; incompressible=false)
const mat_SVK_inc = Hyperelastic(W_SVK, t_HE)

# ---------------------------------------------------------------------------
# Geometry: unit-square Q4/Q9, deformed enough that C₃₃ ≠ its reference value,
# so the plane-stress Newton loop actually iterates rather than converging in
# a single trivial step.
# ---------------------------------------------------------------------------
const X_Q4 = [Vec{3}((0.,0.,0.)), Vec{3}((1.,0.,0.)), Vec{3}((1.,1.,0.)), Vec{3}((0.,1.,0.))]
const X_Q9 = [X_Q4..., Vec{3}((0.5,0.,0.)), Vec{3}((1.,0.5,0.)), Vec{3}((0.5,1.,0.)),
              Vec{3}((0.,0.5,0.)), Vec{3}((0.5,0.5,0.))]

make_q4() = ShellCellValues(QuadratureRule{RefQuadrilateral}(1),
                             Lagrange{RefQuadrilateral,1}(), Lagrange{RefQuadrilateral,1}())
make_q9() = ShellCellValues(QuadratureRule{RefQuadrilateral}(3),
                             Lagrange{RefQuadrilateral,2}(), Lagrange{RefQuadrilateral,2}())

# Non-`const`, non-function-local globals defeat Julia's type inference (every
# closure over them boxes), which shows up as phantom "allocation" that has nothing
# to do with the material model. Layers 1-3 below MUST run inside a function barrier
# — never measure @allocated on a closure over bare top-level globals.
function layer123_report()
    scv4 = make_q4(); reinit!(scv4, X_Q4)
    A     = scv4.A_metric[1]
    A₁, A₂, G₃ = scv4.A₁[1], scv4.A₂[1], scv4.G₃_elem[1]
    c_def = SymmetricTensor{2,2}((1.20, 0.08, 0.90))   # finite in-plane stretch, ≠ reference
    det_A = det(A)
    Jinv  = inv(_J_ref(A₁, A₂, G₃))

    # Layer 1: the plane-stress Newton condensation itself, primal only (no AD
    # nesting) — isolates whether the Newton loop / its closures allocate at all.
    b_c33 = alloc_of(() -> _C33_planestress(mat_SVK_ps, c_def, 0.02, -0.01, det_A, Jinv))

    # Layer 2: membrane_stress_and_tangent — what element assembly calls per QP.
    # Runs the Newton condensation *underneath* a nested Hessian AD, i.e. Dual-of-Dual
    # C₃₃ Newton iterates. Compare incompressible=true (control, no Newton) vs
    # incompressible=false (suspect) for the SAME energy function.
    bytes_inc = alloc_of(() -> membrane_stress_and_tangent(mat_SVK_inc, c_def, A, A₁, A₂, G₃))
    bytes_ps  = alloc_of(() -> membrane_stress_and_tangent(mat_SVK_ps,  c_def, A, A₁, A₂, G₃))

    # Layer 3: bending_and_shear_stiffness — calls membrane_stress_and_tangent again
    # internally plus a shear Hessian, so any Layer-2 allocation compounds.
    bytes_bend_inc = alloc_of(() -> bending_and_shear_stiffness(mat_SVK_inc, c_def, A, A₁, A₂, G₃))
    bytes_bend_ps  = alloc_of(() -> bending_and_shear_stiffness(mat_SVK_ps,  c_def, A, A₁, A₂, G₃))

    (; b_c33, bytes_inc, bytes_ps, bytes_bend_inc, bytes_bend_ps)
end

scv4 = make_q4(); reinit!(scv4, X_Q4)
scv9 = make_q9(); reinit!(scv9, X_Q9)

@testset "Hyperelastic material-model allocations" begin

    r = layer123_report()
    @test r.b_c33 == 0
    @test r.bytes_inc == 0
    @test r.bytes_bend_inc == 0
    println("_C33_planestress (primal, no AD nesting)   : $(r.b_c33) bytes/call")
    println("membrane_stress_and_tangent  incompressible=true : $(r.bytes_inc) bytes/call")
    println("membrane_stress_and_tangent  incompressible=false: $(r.bytes_ps) bytes/call")
    println("bending_and_shear_stiffness  incompressible=true : $(r.bytes_bend_inc) bytes/call")
    println("bending_and_shear_stiffness  incompressible=false: $(r.bytes_bend_ps) bytes/call")

    # --- Layer 4: full element kernels (Q4 and Q9), the actual per-cell assembly
    # calls made during a Newton solve. NH (incompressible) is the known-good
    # baseline from test_allocations.jl; SVK-ps is the reported-bad case. ---
    for (name, scv, n_e) in (("Q4", scv4, 20), ("Q9", scv9, 45))
        u_e = 0.01 .* randn(n_e)
        ke  = zeros(n_e, n_e); re = zeros(n_e)
        for (mname, mat) in (("NH (incompressible)", mat_NH),
                              ("SVK (incompressible)", mat_SVK_inc),
                              ("SVK (plane-stress Newton)", mat_SVK_ps))
            b_re_m = alloc_of(() -> membrane_residuals_RM!(re, scv, u_e, mat))
            b_re_b = alloc_of(() -> bending_residuals_RM!(re, scv, u_e, mat))
            b_ke_m = alloc_of(() -> membrane_tangent_RM!(ke, scv, u_e, mat))
            b_ke_b = alloc_of(() -> bending_tangent_RM!(ke, scv, u_e, mat))
            println("[$name] $mname: residual(mem)=$b_re_m  residual(bend)=$b_re_b  " *
                     "tangent(mem)=$b_ke_m  tangent(bend)=$b_ke_b  bytes/call")
            mat === mat_NH && @test b_re_m == 0 && b_re_b == 0 && b_ke_m == 0 && b_ke_b == 0
        end
    end

    # --- Projection: extrapolate the worst per-call number above to a large-mesh
    # Newton solve, to connect "bytes/call" to the reported OOM. ---
    worst = maximum((r.bytes_ps, r.bytes_bend_ps))
    if worst > 0
        n_elem, n_qp, n_newton = 100_000, 4, 15   # rough "large problem" scale
        total_gb = worst * n_elem * n_qp * n_newton * 4 / 1e9   # ×4: mem+bend, res+tangent calls
        println()
        println("Projected garbage for a $n_elem-element mesh, $n_qp QP/element, " *
                 "$n_newton Newton iterations: ≈ $(round(total_gb, digits=1)) GB " *
                 "(transient, but enough to thrash/exceed RAM before GC catches up).")
    else
        println()
        println("Material-model layer is allocation-free at every level tested " *
                 "(including the plane-stress Newton condensation). The OOM is not " *
                 "coming from membrane_stress_and_tangent / bending_and_shear_stiffness " *
                 "/ _C33_planestress — look at the outer Newton-solve driver instead " *
                 "(sparse K allocation/factorization each iteration, non-const globals " *
                 "in the driver script, or line-search temporaries).")
    end
end
