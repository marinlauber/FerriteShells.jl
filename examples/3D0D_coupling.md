Here's a summary of the architecture and what was done:

---
Operator splitting approach (Lie–Trotter)

At each macro step [t, t+dt_cpl]:

┌───────┬────────────────────────────────────────────────────────┬─────────────────────────────────────────────┐
│ Step│ Action │ API │
├───────┼────────────────────────────────────────────────────────┼─────────────────────────────────────────────┤
│ Fluid │ Advance Windkessel by dt_cpl with Plv fixed│ ODE.step!(integrator, dt_cpl, true) │
├───────┼────────────────────────────────────────────────────────┼─────────────────────────────────────────────┤
│ Solid │ Bordering Newton: find (u, p) s.t. V₃D(u) = Vlv_full/4 │ existing bordering solver │
├───────┼────────────────────────────────────────────────────────┼─────────────────────────────────────────────┤
│ Sync│ Write new p back into integrator.u[3]│ integrator.u[3] = p * Pa2mmHg + u_modified! │
└───────┴────────────────────────────────────────────────────────┴─────────────────────────────────────────────┘

The Windkessel! ODE has du[3] = 0 because Plv is an external input from the solid — it lives in u[3] as a pass-through state, updated after
each Newton solve.

Key points:
- ODE.step!(integrator, dt_cpl, true) takes multiple internal micro-steps (controlled by dtmax=0.001) to reach exactly t + dt_cpl
- ODE.u_modified!(integrator, true) tells the adaptive integrator the state was externally modified, so it resets its error estimate for the
 next step
- The factor of 4 converts between the quarter-domain volume (V₃D) and the full LV volume (Vlv) that the Windkessel tracks

Scale note: The toy geometry (L=1m) has volumes ∼10⁴ ml vs. the cardiac range ∼200 ml. For physiologically meaningful results, either scale
L down (L≈0.05m gives ∼ml-range volumes) or rescale the Windkessel resistances/compliances to match the 3D model's pressure–volume regime.

✻ Cogitated for 8m 6s

❯ So the Lie-Trotter doesn't iterate over the ODE/PDE to find a fixed point?

● Correct — Lie–Trotter does not iterate. It's a first-order splitting: each sub-step sees the other sub-system's state frozen at the previous
 time level, then moves on. No inner loop to enforce consistency.

What this means concretely:

At step t → t+dt:
- ODE uses Plv(t) (old) → produces Vlv(t+dt)
- Newton uses Vlv(t+dt) → produces p(t+dt)
- The ODE step that already ran never saw p(t+dt)

This introduces an O(dt) lag in the coupling. Overall the scheme is first-order in time.

When it fails: when the coupling is stiff — e.g. during valve opening/closing events where a small volume change causes a large pressure
jump. The stale Plv in the ODE step can trigger the wrong valve state.

Alternatives that do iterate:

1. Fixed-point iteration per step (simplest, robust for moderate stiffness):
for coupling_iter in 1:max_coupling_iter
p_prev = p
# re-solve ODE from t with current Plv
# re-solve Newton with new Vlv
abs(p - p_prev) < tol_cpl && break
end
1. Requires resetting the ODE integrator each inner iteration, which is cheap for explicit methods (Tsit5).
2. Godunov splitting (second-order, no extra iterations):
- Half-step ODE with Plv(t)
- Full Newton
- Half-step ODE with p(t+dt)

Doubles the ODE cost but eliminates the O(dt) lag without inner loops.
3. Monolithic (fully implicit, most accurate, most expensive): augment the Newton system with the ODE residual — the ODE and PDE are solved
simultaneously. This is what Thunderbolt's SchurComplementLinearSolver is doing: the ODE states and the LV pressure are extra unknowns in
the same Newton system.

For a haemodynamic model where the cardiac cycle involves valve events, the fixed-point iteration approach (option 1) is usually the
pragmatic middle ground — it's what most research codes use before going fully monolithic.

✻ Cooked for 33s

❯ so what is done exactly in @../Thunderbolt.jl/bak/examples/lv-with-mtk-circuit.jl ?
⎿  Read ../Thunderbolt.jl/bak/examples/lv-with-mtk-circuit.jl (233 lines)

● Let me re-read the Thunderbolt example carefully to give a precise answer.

The key lines are:

chamber_solver = HomotopyPathSolver(
NewtonRaphsonSolver(;
inner_solver=SchurComplementLinearSolver(
LinearSolve.UMFPACKFactorization()
)
)
)
blood_circuit_solver = Tsit5()
timestepper = LieTrotterGodunov((chamber_solver, blood_circuit_solver))

It's also operator splitting — not monolithic. LieTrotterGodunov alternates between the two sub-systems without iterating between them to
find a fixed point. The name suggests a symmetric (Godunov/Strang) variant:

solid(dt/2) → circuit(dt) → solid(dt/2)

which is second-order in time, vs our first-order solid(dt) → circuit(dt).

What each piece does:

┌─────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│Component│ Role │
├─────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ LieTrotterGodunov │ Splits 3D solid ↔ 0D circuit in time; no cross-iteration │
├─────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ HomotopyPathSolver│ Handles the nonlinear solid step with path-following (automatic load sub-stepping if Newton stalls)│
├─────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ SchurComplementLinearSolver │ Solves the augmented linear system inside each Newton iteration — this is exactly the bordering formula│
│ │ we implemented manually: two back-substitutions + scalar Schur complement for (δu, δp) │
├─────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Tsit5() │ Advances the 0D circuit ODE │
└─────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────┘

So SchurComplementLinearSolver is not coupling the 3D and 0D models — it's the bordered linear solver within the 3D solid Newton step for the (u, p) pair given a fixed V_target from the circuit. That's exactly S = -dot(dVdu, v2), δp = (...)/ S in our code.

Our 3D0D.jl is doing the same thing, just with:
- Manual lu!/ldiv! instead of SchurComplementLinearSolver
- Simple Lie-Trotter (1st order) instead of the symmetric Godunov variant (2nd order)
- No HomotopyPathSolver (we rely on small enough dt_cpl so Newton doesn't need sub-stepping)

To upgrade to the Godunov symmetric split you'd just run the Newton solve at the start and end of each step at half-dt, which doubles the solid cost but eliminates the O(dt) coupling lag.

## Newton convergence

With a good initial guess (previous step's solution as warm start), the initial residual after applying the new BCs is O(Δλ). Each Newton
iteration squares the error:

‖R₀‖ ~ Δλ  (after applying new BCs, free DOFs haven't adjusted)
‖R₁‖ ~ ‖R₀‖²  (first correction)
‖R₂‖ ~ ‖R₁‖² = ‖R₀‖⁴  (second correction — usually below tol)

With tol = 1e-6 and a step size that gives ‖R₀‖ ~ 1e-2:
- Iter 1: ‖R‖ ~ 1e-4 → not converged
- Iter 2: ‖R‖ ~ 1e-8 → converged

That's 2 solves + 1 final check = 3 assembly evaluations, which is what newton_itr counts.

You see this across the RM examples (RollupCantilever, SquareAirbag, now LIMO) because they all sit in the same regime: step size small enough that ‖R₀‖ is moderate (~1e-2 to 1e-3), tolerance at 1e-6, quadratic convergence → always lands in 2–3 iterations. If you halve the step size, ‖R₀‖ halves, and you'd still get 3 iterations (just the final residual is much smaller after iter 2). You'd only drop to 2 iterations if ‖R₀‖ were already near √tol ~ 1e-3.

Quadratic convergence is a property of the exact Newton method — if the tangent K is the true derivative of the residual, you get quadratic convergence. Any deviation means the tangent is approximate:

- More iterations (4–6), residual decreasing but slowly: tangent is slightly wrong — common causes are ForwardDiff truncation, missing geometric stiffness terms, or symmetry approximations (K_IJ ≠ K_JI filled incorrectly).
- Stagnation at some residual floor: tangent is consistent but there's a term missing entirely (e.g. missing follower-pressure stiffness
K_pres in a pressure-loaded problem). The solution drifts to a slightly wrong equilibrium.
- Linear convergence (residual halves each step): the "tangent" is actually a fixed matrix reused across iterations (e.g. initial stiffness method), not updated each step.
- Divergence: tangent has wrong sign somewhere, or a term dominates with the wrong sign (like the unit-inconsistent Pact from earlier — K_eff = K_int - (p - Pact_in_mmHg)*K_pres made K_eff indefinite).

In your RM explicit tangent (bending_tangent_RM_explicit!), the most likely source of imperfection is the φφ geometric block — it has a ∂²d/∂φ∂φ term that's only assembled on diagonal J=I blocks. If that term is slightly off, you'd see 4–5 iterations instead of 3 on bending-dominated problems, but it wouldn't diverge. You can always verify by replacing the explicit tangent temporarily with a ForwardDiff Jacobian of the residual and checking if the iteration count drops back to 3.

# Two RHS per iteration: residual and reference load direction
v1 = K_free \ (-r_free)          # standard Newton step
v2 = K_free \ f_ref_free         # tangent to λ direction

# arc-length constraint gives δλ (cylindrical)
δλ = (-dot(Δu_free, v1)) / (dot(Δu_free, v2) + ψ² * Δλ)
δu = v1 + δλ * v2

Δu_free .+= δu; Δλ += δλ

The tricky part is computing f_ref_free. For your case (prescribed displacements scaling with λ), the cleanest approach is a finite difference:

ε = 1e-6
update!(ch, λ + ε); u_pert = copy(u); apply!(u_pert, ch)
assemble_all!(K_tmp, r_pert, dh, scv, u_pert, mat)
apply_zero!(K_tmp, r_pert, ch)
f_ref = (r_pert .- r_int) ./ ε   # only free DOFs matter
update!(ch, λ)                    # restore


The predictor (first step from a converged point) is just the tangent direction normalized to Δs:

v_pred = K_free \ f_ref_free
Δλ_pred = Δs / sqrt(dot(v_pred, v_pred) + ψ²)
Δu_pred = Δλ_pred * v_pred

The sign of Δλ_pred should follow the previous step to stay on the same branch.

ψ scales the relative weight of λ vs displacement in the arc-length — ψ = 0 gives cylindrical (displacement only), ψ = 1 gives spherical. Start with ψ = 0 since
λ and displacements have very different magnitudes here.