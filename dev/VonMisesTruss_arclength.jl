# Von Mises (double) truss snap-through — Riks arc-length continuation
#
# Geometry:
#   Node 1 (-a, 0)  pinned
#   Node 2 ( a, 0)  pinned
#   Node 3 ( 0, h)  free (crown)
#
# DOFs: node k → [2k-1, 2k]  (x, y)
#   dof 1,2 = node 1  (fixed)
#   dof 3,4 = node 2  (fixed)
#   dof 5,6 = node 3  (free)
#
# Load: unit downward force at crown → f_ref[6] = -1
#
# The load factor λ parameterises the applied load: F_ext = λ * f_ref.
# Arc-length traces the full load-displacement path through the snap-through.

using LinearAlgebra, Printf

# ── Parameters ──────────────────────────────────────────────────────────────
const a  = 1.0       # half-span
const h  = 0.1       # rise (shallow truss for snap-through)
const EA = 1000.0    # axial stiffness

const NDOF  = 6
const FREE  = [5, 6]              # free DOFs (crown x and y)
const FIXED = [1, 2, 3, 4]       # pinned DOFs

# Reference node positions
const X = [-a  0.0;   # node 1
            a  0.0;   # node 2
            0.0  h]   # node 3 (crown)

# Element connectivity: (node_i, node_j)
const ELEMENTS = [(1, 3), (2, 3)]

# ── Truss element assembly ───────────────────────────────────────────────────
# Returns the 4×4 tangent stiffness and 4-vector internal force
# for a geometrically nonlinear truss element connecting nodes i and j.
# Local DOF order: [u_xi, u_yi, u_xj, u_yj]
function element_stiffness(Xi, Xj, ui, uj)
    # Current positions
    xi = Xi + ui
    xj = Xj + uj

    # Reference length
    d0 = Xj - Xi
    L0 = norm(d0)

    # Current length and unit vector
    d  = xj - xi
    L  = norm(d)
    ev = d / L            # unit tangent vector

    # Engineering strain and axial force
    ε = (L - L0) / L0
    N = EA * ε

    # b = [-ev; ev] (4-vector)
    b = [-ev[1], -ev[2], ev[1], ev[2]]

    # Qe: (1/L)*[I -I; -I I] — the rank-2 projection onto the plane ⊥ ev
    # geometrically: Qe - b*b' removes the axial direction
    Qe = (1/L) * [1 0 -1 0;
                  0 1  0 -1;
                 -1 0  1  0;
                  0 -1 0  1]

    # Material stiffness (along bar) + geometric stiffness (transverse)
    Ke = (EA/L0) * (b * b') + (N/L) * (Qe - b * b')

    # Internal force
    fe = N * b

    return Ke, fe
end

# ── Global assembly ──────────────────────────────────────────────────────────
function assemble(u)
    K = zeros(NDOF, NDOF)
    f = zeros(NDOF)

    for (ni, nj) in ELEMENTS
        # Global DOF indices for this element
        dofs = [2ni-1, 2ni, 2nj-1, 2nj]

        # Node positions and displacements
        Xi = X[ni, :]
        Xj = X[nj, :]
        ui = u[2ni-1:2ni]
        uj = u[2nj-1:2nj]

        Ke, fe = element_stiffness(Xi, Xj, ui, uj)

        for (a, dof_a) in enumerate(dofs)
            f[dof_a] += fe[a]
            for (b, dof_b) in enumerate(dofs)
                K[dof_a, dof_b] += Ke[a, b]
            end
        end
    end

    return K, f
end

# ── Boundary conditions ──────────────────────────────────────────────────────
# Apply pinned BCs by zeroing rows/cols and setting diagonal = 1
function apply_bc!(K, rhs)
    for d in FIXED
        K[d, :] .= 0.0
        K[:, d] .= 0.0
        K[d, d]  = 1.0
        rhs[d]   = 0.0
    end
end

# ── Reference load vector ────────────────────────────────────────────────────
const f_ref = zeros(NDOF)
f_ref[6] = -1.0    # unit downward force at crown (dof 6 = y of node 3)

# ── Arc-length solver (Riks, cylindrical, ψ = 0) ────────────────────────────
#
# Constraint:  ‖Δu[FREE]‖² = Δs²
#
# Predictor step:
#   Solve  K * v = f_ref  (with BCs)
#   Scale: δλ_pred = ±Δs / ‖v[FREE]‖
#   Δu = δλ_pred * v,  Δλ = δλ_pred
#
# Corrector iterations:
#   Residual: g = f_int(u) − λ * f_ref
#   Solve    K * v1 = −g
#   Solve    K * v2 =  f_ref
#   Arc-length equation (differentiated):
#     dot(Δu[FREE], Δu[FREE] + δλ*v2[FREE]) + dot(Δu[FREE], v1[FREE]) = 0
#     δλ = −dot(Δu[FREE], v1[FREE]) / dot(Δu[FREE], v2[FREE])
#   Update:  Δu[FREE] += v1[FREE] + δλ * v2[FREE]
#            Δλ       += δλ
#   Accept when ‖g[FREE]‖ < tol

function solve_arclength(;
    Δs_init  = 5e-4,    # initial arc-length step
    max_step = 200,     # maximum arc-length steps
    max_cor  = 20,      # maximum corrector iterations
    tol      = 1e-10,   # convergence tolerance on ‖g[FREE]‖
    verbose  = true,
)
    u  = zeros(NDOF)    # displacement (starts from zero)
    λ  = 0.0            # load factor

    Δu      = zeros(NDOF)   # current increment (accumulated over correctors)
    Δλ      = 0.0
    Δs      = Δs_init

    # Track previous converged increment for sign control
    Δu_prev = zeros(NDOF)
    Δλ_prev = 0.0

    # Storage for results
    results = [(λ, u[6])]   # (load factor, crown y-displacement)

    verbose && @printf("%-6s  %-10s  %-12s  %-8s\n", "step", "λ", "u_crown", "iters")

    for step in 1:max_step

        # ── Predictor ──────────────────────────────────────────────────────
        K, _ = assemble(u)
        rhs  = copy(f_ref)
        apply_bc!(K, rhs)
        v    = K \ rhs          # tangent to load-displacement path

        vn   = norm(v[FREE])
        vn < 1e-15 && error("Singular predictor at step $step")

        # Sign control: choose direction consistent with previous step
        δλ_pred = Δs / vn
        if step > 1
            dot(Δu_prev[FREE], v[FREE]) + Δλ_prev * δλ_pred < 0 && (δλ_pred = -δλ_pred)
        end

        # corrections
        Δu      .= δλ_pred .* v
        Δλ       = δλ_pred
        converged = false

        # ── Corrector ──────────────────────────────────────────────────────
        iters = 0
        for cor in 1:max_cor
            iters = cor
            u_trial = u .+ Δu

            K, f_int = assemble(u_trial)
            g = f_int .- (λ + Δλ) .* f_ref    # equilibrium residual

            normg = norm(g[FREE])
            if normg < tol
                converged = true
                break
            end

            # Apply BCs to the two RHS vectors
            rhs1 = -g
            rhs2 = copy(f_ref)
            apply_bc!(K, rhs1)    # modifies K in-place (but K is recomputed each iter)
            # K2 = copy(K)          # need fresh K for second solve TODO not
            # apply_bc!(K2, rhs2)

            v1 = K  \ rhs1        # displacement correction
            v2 = K  \ rhs2        # load sensitivity direction

            # Arc-length corrector equation:
            #   dot(Δu_free, Δu_free + v1_free + δλ*(v2_free)) = Δs²
            # Linearised (omitting ‖v1‖² term):
            #   δλ = -dot(Δu_free, v1_free) / dot(Δu_free, v2_free)
            denom = dot(Δu[FREE], v2[FREE])
            abs(denom) < 1e-15 && error("Singular arc-length system at step $step, cor $cor")
            δλ = -dot(Δu[FREE], v1[FREE]) / denom

            Δu[FREE] .+= v1[FREE] .+ δλ .* v2[FREE]
            Δλ        += δλ
        end

        if !converged
            @warn "Step $step did not converge in $max_cor correctors; halving Δs"
            Δs /= 2
            continue
        end

        # ── Accept step ────────────────────────────────────────────────────
        u          .+= Δu
        λ           += Δλ
        Δu_prev    .= Δu
        Δλ_prev     = Δλ

        push!(results, (λ, u[6]))

        verbose && @printf("%-6d  %-10.4f  %-12.6f  %-8d\n", step, λ, u[6], iters)

        # Terminate after the full path is traced:
        # crown passes through inverted stable state (current y = -h, i.e. u[6] < -2h)
        u[6] < -2h - 0.01 && break
    end

    return results
end

# ── Analytical snap-through load ─────────────────────────────────────────────
# For a symmetric shallow von Mises truss the snap-through load is:
#   λ_max = (EA * h) / L0 * (1 - (h²/3 + a²/L0²) * (L0/h) * ...)
# Simpler: λ_max = 2*EA*h/L0 * (ε_c) where ε_c is the critical strain.
# Exact critical point: vertical equilibrium N sin(θ) * 2 = λ,
# at v_c = h/√3 (crown drops to height h - h/√3 = h(1-1/√3)).
function analytical_snap_through()
    L0 = hypot(a, h)
    θ0 = atan(h, a)
    # At v_c = h*(1 - 1/√3), the horizontal projection stays a, vertical is h/√3
    vc = h * (1 - 1/sqrt(3))
    Lc = hypot(a, h - vc)
    εc = (Lc - L0) / L0
    Nc = EA * εc
    θc = atan((h - vc), a)
    # Equilibrium: 2*N*(h-vc)/Lc = -λ (f_ref[6]=-1 convention), so λ = -2*Nc*sin(θc)
    λc = -2 * Nc * sin(θc)
    return λc, vc
end

# ── Run ──────────────────────────────────────────────────────────────────────
λ_snap, v_snap = analytical_snap_through()
println("Analytical snap-through:  λ_max ≈ $(round(λ_snap, digits=4)) N  at  v_crown ≈ $(round(v_snap, sigdigits=4)) m\n")

results = solve_arclength(Δs_init=5e-4, max_step=500, tol=1e-10, verbose=true)

println("\nDone. $(length(results)) points traced.")
λs = first.(results)
us = last.(results)
idx_max = argmax(λs)
idx_min = argmin(λs)
println("First  limit point:  λ_max = $(round(λs[idx_max], digits=4))  at  u_crown = $(round(us[idx_max], sigdigits=4))  (snap-through)")
println("Second limit point:  λ_min = $(round(λs[idx_min], digits=4))  at  u_crown = $(round(us[idx_min], sigdigits=4))  (snap-back, inverted)")
