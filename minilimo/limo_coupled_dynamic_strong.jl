using FerriteShells, LinearAlgebra, Printf, WriteVTK, QuadGK, JLD2
include(joinpath(@__DIR__, "util.jl"))

# Strongly (monolithically) coupled dynamic miniLIMO: HHT-α structure + 0D Windkessel
# solved together in ONE Newton system per time step.  Counterpart of the weakly-coupled
# `limo_coupled_dynamic_weak.jl` (Lie–Trotter split with a black-box ODE integrator).
#
#   PHASE 1 — dynamic morph  (t ∈ [0, T_sim], HHT-α)   [identical to the transient file]
#     Morph the edge onto the elliptic arc + fill Plv → morphed state u = un, v, a, Plv=p_max.
#
#   PHASE 2 — monolithic strong coupling  (HHT-α, Δt = dt_cpl)
#     The Windkessel is discretized by implicit Euler and embedded in the structural Newton;
#     the black-box ODE integrator is gone.  The LV chamber volume is NOT an independent 0D
#     state — it is V_LV(u) = 2·V₃D(u); Plv is the pressure that closes the flow balance.
#     Unknowns per step: (u ∈ Rᴺ, Plv, Pa, Pv).  Residuals:
#       R_s(u,Plv) = M·ä + (1−α)(α_damp·Mv + r_int − F_ext(u,Plv,Pact)) + α·g_old   [N]
#       R_vol      = (V_LV(u) − V_LVₙ)/Δt − (Qmv − Qao)                              [1]
#       R_art      = (Pa − Paₙ)/Δt − (Qao/Ca + (Pv − Pa)/(Rp·Ca))                    [1]
#       R_ven      = (Pv − Pvₙ)/Δt − ((Pa − Pv)/(Rp·Cv) − Qmv/Cv)                    [1]
#     with F_ext = Plv·F_plv + Pact·F_pact − Pact·F_plvpact, V_LV = 2·V₃D = −2·compute_volume,
#     valves Qmv(Plv,Pv), Qao(Plv,Pa) (diodes).  Bordered/condensed Newton:
#       v1 = K_eff⁻¹(−R_s),  v2 = (1−α)·K_eff⁻¹F_plv,  δu = v1 + δPlv·v2,
#       aᵀ = (1/Δt)·∂V_LV/∂u = −(2/Δt)·dVdu  (dVdu = ∂compute_volume/∂u = −∂V₃D/∂u),
#       3×3 solve for (δPlv, δPa, δPv) with the volume row augmented by aᵀv2 / aᵀv1.
#     Everything in SI (Pa, m³, s); pressures reported in mmHg, volumes in ml.
#     Three follower surfaces: SRF_1 Plv, SRF_2 Plv−Pact, SRF_3 Pact.

# Valve diodes (SI: Pa, m³/s).  Return (Q, ∂Q/∂P_first, ∂Q/∂P_second) on the active branch.
# Mitral: venous → LV filling, opens when Pv ≥ Plv.  Args (Plv, Pv).
@inline function mitral_flow(Plv, Pv, Rv; R_closed=1e10)
    Pv ≥ Plv ? ((Pv - Plv)/Rv,       -1/Rv,       1/Rv) :
               ((Plv - Pv)/R_closed,  1/R_closed, -1/R_closed)   # (Q, ∂/∂Plv, ∂/∂Pv)
end
# Aortic: LV → arterial ejection, opens when Plv ≥ Pa.  Args (Plv, Pa).
@inline function aortic_flow(Plv, Pa, Ra; R_closed=1e10)
    Plv ≥ Pa ? ((Plv - Pa)/Ra,        1/Ra,       -1/Ra) :
               ((Pa - Plv)/R_closed, -1/R_closed,  1/R_closed)   # (Q, ∂/∂Plv, ∂/∂Pa)
end

# material
ρ   = 1200.0       # density [kg/m³]
E = 20e6
ν = 0.3
thickness = 0.001
# mat = LinearElastic(0.35e9, 0.3, 0.0002) # nylon-cpated TPU
# mat = LinearElastic(E, ν, thickness) # soft TPU

# Saint-Venant-Kirchhoff material, W(C) = λ/2 tr(E)² + μ E:E, E = (C-I)/2.  λ, μ are
# struct fields (not closed-over module-level globals) so the energy call is
# type-stable — a non-const-global capture here breaks type inference through the
# nested ForwardDiff Hessian and the plane-stress Newton condensation
# (incompressible=false), turning each material call into ~17KB of garbage instead
# of 0 (see test/test_hyperelastic_allocations.jl).
struct SVKEnergy{T}
    λ::T
    μ::T
end
(w::SVKEnergy)(C) = (Eg = (C - one(C))/2; w.λ/2 * tr(Eg)^2 + w.μ * (Eg ⊡ Eg))

# NOTE: do not add an `SVKEnergy(E, ν)` outer constructor for (E,ν) → (λ,μ) — with
# both args the same concrete type, Julia's auto-generated default constructor
# `SVKEnergy(λ::T, μ::T) where T` is more specific and silently wins over it,
# so E, ν would be stored verbatim as λ, μ instead of being converted.
λ_svk = E*ν/((1+ν)*(1-2ν))
μ_svk = E/(2*(1+ν))
# mat = Hyperelastic(SVKEnergy(λ_svk, μ_svk), thickness; incompressible=false)

# WRINKLING (tension-field) MATERIAL.  A 1 mm TPU sheet cannot carry membrane
# compression: where the minor principal resultant goes negative the sheet
# physically wrinkles.  A non-relaxed model has no way to express that, so it
# represents it as a mesh-scale fold instead — which is what drives K_eff
# indefinite in Phase 2.  `tension_field=true` applies the Roddeman relaxation
# (uniaxial tension along the major axis when σ₂ < 0, slack when both ≤ 0), shedding
# the compression rather than buckling on it.  Bending/shear stiffness is left
# UN-relaxed (src/material.jl:112-121), so the rotational block keeps full stiffness.
# `ε_tf` is the positive-definiteness floor on the relaxed tangent: raise it (1e-2)
# if Newton stalls chattering on the taut↔wrinkled switch, lower it (1e-4) for a
# sharper wrinkle.  Same material as limo_full_coupled_dynamic_strong.jl:36.
#
# NOTE: `Hyperelastic` has NO tension-field path (src/material.jl:276-287) — moving
# to SVK silently removed this relaxation.  SVK ≈ LinearElastic at these strains
# anyway; the large-rotation kinematics live in the shell, not the constitutive law.
#
# BENDING-SCALE CONTINUATION (β).  `β` multiplies the bending stiffness D and the
# transverse-shear stiffness while leaving the MEMBRANE stiffness untouched
# (src/material.jl:112-121).  β>1 is an artificial bending stabilization: it raises the
# critical buckling load ∝β and lengthens the wrinkle half-wavelength ∝β^(1/4), so the
# mesh resolves what would otherwise collapse into a mesh-scale fold.
#
# Unlike damping, continuation has an EXIT.  β is walked back to 1 during the first beat
# (see `β_cont` next to the Phase-2 controls), so every cycle after the first — and in
# particular the last one, the only one whose geometry is saved — runs on the PHYSICAL
# shell.  That also makes it a diagnostic: if the fold reappears as β→1, the instability
# is real and the geometry needs changing; if it does not, it was a numerical artifact.
#
# β_start applies from t=0 (Phase 1) so there is no material jump at the phase boundary.
# The morph is displacement-controlled, so a stiffer bending block changes the morphed
# SHAPE very little — it just makes the fold smoother.  β_start=1.0 disables continuation
# entirely and recovers the previous behaviour.
#
# TENSION FIELD IS OFF — deliberately.  It removes the buckling, but at a cost that is
# fatal here.  In `tension_field_relax` (src/material.jl:59-75) the SLACK branch returns
# `ε·Ñ`: the whole stress scaled by ε_tf, so a slack region keeps only ε_tf (=0.1%) of its
# in-plane stiffness IN EVERY DIRECTION.  Measured tangent eigenvalues, slack vs taut:
# [13.3, 16.4, 37.4] vs [13260, 16350, 37430] — exactly 1e-3×.  Bending cannot make up for
# it: bending resists CURVATURE, while element stretch and shear are resisted only by the
# membrane term, so nothing holds element shape and the mesh distorts wherever relaxation
# is active.  That is intrinsic to relaxed-energy models — the relaxed energy is not
# strictly convex, so the deformation in the relaxed direction is a true null mode; a real
# membrane selects a wrinkle amplitude through bending, which is precisely the information
# tension field throws away.
#
# It also corrupts the result, not just the picture: `compute_volume` integrates over the
# endocardium, so a distorted endocardium feeds a wrong V_LV into the Windkessel.
#
# The β route below is the well-posed alternative to the same problem: instead of deleting
# the compression, keep it and make the buckling wavelength long enough for the mesh to
# resolve.  The tangent stays positive definite and the mesh stays clean.
# To experiment with partial relaxation anyway, add `tension_field=true, ε_tf=0.1` — a much
# larger floor than 1e-3 trades less compression-shedding for far less distortion.
mat_β(β) = LinearElastic(E, ν, thickness; β=β)
β_start = 4.0
β_end   = 1.0   # β to settle at; raise above 1 to keep a permanent bending stiffener
mat = mat_β(β_start)

@show mat
Np = 3
# NEW GEOMETRY (x_act=0.044), with the skirt RESOLVED.  Same geometry as before —
# only the element counts changed.  Two independent reasons the 4/4/33 counts break
# Phase 2 on this geometry, both absent on the old x_act=0.035 one:
#
#  (a) The skirt between the actuator weld and the symmetry clamp collapsed from
#      15.59 mm (x) / 19.00 mm (y) to 6.59 / 7.00 mm.  The shell edge boundary layer
#      is √(R·t) ≈ 5.5 mm here, so the weld-kink layer and the clamp layer now
#      OVERLAP inside a band that must absorb the whole 80 kPa pouch expansion.  That
#      band goes into compression and buckles.  nx_left/right 4→8 and ny_top 4→8 put
#      ~8 elements across the boundary layer instead of ~1.2.
#
#  (b) Follower-pressure softening.  `−Pact·K_pact` is a DESTABILIZING contribution to
#      K_eff scaling like Pact·h, while the bending stiffness it must not overcome
#      scales like D/h² — so the ratio grows as h³.  x_act 0.035→0.044 with nx_act
#      only 30→33 grew the actuator element 2.33→2.67 mm, pushing that ratio from
#      ~0.55 to ~0.83 at peak Pact=600 mmHg.  In the rotational block, which carries
#      NO mass regularization (see the rotary-inertia note at `assemble_mass!`), 0.83
#      is enough on its own to flip the tangent indefinite.  nx_act 33→42 brings the
#      actuator element back to 2.10 mm → ratio ≈ 0.40.  (42 is divisible by Np=3.)
#
# Cost: 5844 cells vs 4126 (~42% more).  If that is too slow, (b) alone — nx_act
# 33→42, leaving the skirt counts at 4 — is the cheaper half; try it first.
grid = make_minilimo_grid(;
    nx_left=8, nx_act=42, nx_right=8,
    ny_bot=10, ny_act=48, ny_top=8,
    W=0.10118, H=0.109, x_act=0.044, y_lo=0.02, y_hi=0.102,
    Np=Np, order=1
)
# Original (under-resolved) counts for this geometry — the ones that buckle:
# grid = make_minilimo_grid(;
#     nx_left=4, nx_act=33, nx_right=4,
#     ny_bot=10, ny_act=48, ny_top=4,
#     W=0.10118, H=0.109, x_act=0.044, y_lo=0.02, y_hi=0.102,
#     Np=Np, order=1
# )
# Old geometry (x_act=0.035) — the configuration that runs today:
# grid = make_minilimo_grid(;
#     nx_left=3*3, nx_act=3*10, nx_right=3*3,
#     ny_bot=3*1, ny_act=3*14, ny_top=3*2,
#     W=0.10118, H=0.109, x_act=0.035, y_lo=0.004, y_hi=0.09,
#     Np=Np, order=1
# )

ip  = Lagrange{RefQuadrilateral, 1}()
qr  = QuadratureRule{RefQuadrilateral}(2)
scv = ShellCellValues(qr, ip, ip; mitc=MITC4)

dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

# The three follower-pressure surfaces (used in the coupled phase; Phase 1 uses Plv_srf).
Plv_srf     = getcellset(grid, "SRF_1") ∪ getcellset(grid, "SRF_2")  # Plv acts here
Pact_srf    = getcellset(grid, "SRF_3")                              # +Pact
PlvPact_srf = getcellset(grid, "SRF_2")                              # −Pact (opposes Plv)

# Smooth sinusoidal ramp: λ(t) = ½(1 − cos(πt/T_morph)) for t ≤ T_morph, 1 beyond.
T_morph = 2.0   # morphing duration [s]
T_sim   = 2.0   # Phase-1 simulation duration [s]
Δt      = 0.001 # initial time step [s]
ramp(t) = t < T_morph ? 0.5 * (1 - cos(π * t / T_morph)) : 1.0

# corner_relief tapers the morph to zero over the first/last 3 edge nodes so the
# edge∩sym corner carries membrane tension instead of the fold-induced compression
# singularity that snaps in Phase 2 (see util.jl; width is mesh/material-sensitive).
prescribed_u = generate_boundary_function(grid, "edge"; ramp=ramp)

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getnodeset(grid, "edge"), (x,t) -> prescribed_u(x, t), [1,3]))
add!(ch, Dirichlet(:u, getnodeset(grid, "edge"), x -> 0.0, [2]))
add!(ch, Dirichlet(:θ, getnodeset(grid, "edge"), x -> zeros(2), [1,2]))
add!(ch, Dirichlet(:u, getfacetset(grid, "sym"), x -> 0.0, [3]))
add!(ch, Dirichlet(:θ, getfacetset(grid, "sym"), x -> zeros(2), [1,2]))
close!(ch); Ferrite.update!(ch, 0.0)

N_dof = ndofs(dh)
free  = ch.free_dofs

# HHT-α parameters  (α = −0.3: strong high-frequency damping, still stable)
α_hht   = -0.3
γ_hht   = 0.5 - α_hht
β_hht   = (1 - α_hht)^2 / 4
α_damp  = 10.0    # mass-proportional Rayleigh damping coefficient [1/s]
# Stiffness-proportional (Rayleigh β) damping, C = α_damp·M + β_damp·K₀.
#
# WHY: the two Rayleigh branches are opposite spectral filters — ζᵢ = α_damp/(2ωᵢ)
# for the mass branch, ζᵢ = β_damp·ωᵢ/2 for the stiffness branch.  Mass damping is a
# LOW-pass damper: α_damp=10 gives ζ≈0.8 on the 1 Hz beat but ζ≈0.005 at 160 Hz, so
# raising it mostly drags on the global motion (forcing small Δt) while leaving the
# short-wavelength fold chatter essentially undamped.  That is why cranking α_damp
# has not helped.  Buckling/folding is a HIGH-frequency, short-wavelength event, so
# it needs the stiffness branch.
#
# Second reason it is the right lever here: the mass matrix carries NO rotary
# inertia (src/assembly.jl:787-799 fills only dofs 5I-4..5I-2), so α_damp·M·v is
# IDENTICALLY ZERO on the director-rotation dofs — the exact block that goes
# indefinite.  β_damp·K₀·v is not; K₀ has full rotational content.
#
# SIZING: this enters K_eff as `cfac·K₀` with cfac = β_damp·(1−α)γ/(βΔt), i.e. a
# K-shaped SPD augmentation of exactly that fraction of K₀.  At Δt=1e-3 the bracket
# is ≈2462, so β_damp=2e-5 → cfac ≈ 0.049, a ~4.9% stiffness augmentation.  That is
# sized deliberately: the tension-field membrane tangent on the WRINKLED branch is
# indefinite by about −2.4% of its largest eigenvalue (FD-verified), so a ~5%
# SPD augmentation comfortably covers it.  Modal effect: ζ=0.05 at ~800 Hz, ζ=0.5 at
# ~8 kHz, ζ=6e-4 on the 1 Hz beat — physical dynamics untouched.  Note cfac ∝ 1/Δt,
# so the augmentation falls to ~1% at Δt_max=5e-3; if Phase 2 only misbehaves at the
# grown Δt, cap Δt_max rather than raising β_damp.  β_damp=0.0 recovers the old runs.
β_damp  = 2.0e-5  # stiffness-proportional Rayleigh damping coefficient [s]
tol      = 1e-4
max_iter = 50
Δt_min   = 1e-7
Δt_max   = 0.005 # careful with this value, can skip interesting details

# Pressure ramp: same sinusoidal profile as morphing, up to p_max [Pa]
Pa2mmHg = 0.00750062 # Pa/mmHg
m3_to_ml = 1.0e6     # m³ → ml
p_max   = 6.0 / Pa2mmHg   # 6 mmHg → Pa

K_int     = allocate_matrix(dh)
K_eff     = allocate_matrix(dh)
K_plv     = allocate_matrix(dh)
K_pact    = allocate_matrix(dh)
K_plvpact = allocate_matrix(dh)
M         = allocate_matrix(dh)
r_int     = zeros(N_dof)
F_plv     = zeros(N_dof)
F_pact    = zeros(N_dof)
F_plvpact = zeros(N_dof)
g_old     = zeros(N_dof)
res       = zeros(N_dof)
δu        = zeros(N_dof)
u_trial   = zeros(N_dof)
rhs       = zeros(N_dof)
v1        = zeros(N_dof)   # coupled-phase Schur vectors
v2        = zeros(N_dof)
dVdu      = zeros(N_dof)
a_new     = zeros(N_dof)
v_new     = zeros(N_dof)
Ma        = zeros(N_dof)
Mv        = zeros(N_dof)
Kv        = zeros(N_dof)   # K₀·v — stiffness-proportional (Rayleigh β) damping force
ũ         = zeros(N_dof)
ṽ         = zeros(N_dof)
u_new     = zeros(N_dof)

# Precomputed shell-DOF maps (fixed for the run) and reusable element buffers.
n_e   = ndofs_per_cell(dh)
ke    = zeros(n_e, n_e)
re    = zeros(n_e)
u_e   = zeros(n_e)
sdofs = Vector{Vector{Int}}(undef, Ferrite.getncells(grid))
_is_rot = falses(N_dof)
for cell in CellIterator(dh)
    sd = shelldofs(cell)
    sdofs[Ferrite.cellid(cell)] = sd
    for I in 1:length(cell.nodes); _is_rot[sd[5I-1]] = true; _is_rot[sd[5I]] = true; end
end
# Split dof index sets so the Newton step cap can use physically meaningful limits:
# translations are metres, director rotations are radians, and the fold rotations run
# 10-100× larger numerically — one shared ∞-norm cap would be set by the rotations and
# would over-restrict the displacements.  PRESCRIBED dofs are excluded: `apply_zero!`
# leaves a unit diagonal there, so v2 = K_eff⁻¹F_plv holds v2[p] = F_plv[p], a FORCE,
# not a displacement.  Those entries are overwritten by `apply!(u_new, ch)` anyway, but
# left in they would set the ∞-norm and throttle every step for no reason.  (Same
# reason the volume row already does `dVdu[ch.prescribed_dofs] .= 0.0`.)
_free = falses(N_dof); _free[free] .= true
rot_dofs = findall(_is_rot .& _free)
trn_dofs = findall((.!_is_rot) .& _free)

assemble_mass!(M, dh, scv, ρ, mat)

m_fac(Δt) = 1 / (β_hht * Δt^2) + (1 - α_hht) * α_damp * γ_hht / (β_hht * Δt)

# Reference tangent for Rayleigh damping, frozen at the undeformed configuration.
# Standard practice in large-deformation dynamics: keeping C constant and SPD avoids
# differentiating K and avoids inheriting the indefiniteness we are damping against.
# Assembled with the PHYSICAL material (β=1), NOT the continuation start: otherwise K₀
# would carry β_start× the bending stiffness and the ~4.9% β_damp sizing documented
# above would silently be ~4.9%·β_start on the rotational block.
assemble_all!(K_int, r_int, dh, scv, zeros(N_dof), mat_β(1.0), sdofs, ke, re, u_e)
K0 = copy(K_int)
assemble_all!(K_int, r_int, dh, scv, zeros(N_dof), mat, sdofs, ke, re, u_e)
K_eff.nzval .= M.nzval .* m_fac(Δt) .+ (1 - α_hht) .* K_int.nzval
apply_zero!(K_eff, r_int, ch)
F_lu = lu(K_eff)

bufs_morph = (; K_int, r_int, K_plv, F_plv, M, K_eff, res, rhs, δu, u_trial, a_new, v_new,
                Ma, Mv, K0, Kv, F_lu, free, g_old, sdofs, ke, re, u_e,
                α_hht, γ_hht, β_hht, α_damp, β_damp)

# HHT-α Newton corrector (with backtracking line search) for one morph time step.
# `u_new` is updated in place; returns (converged, iters). Plv pressure acts on `Plv_srf`.
function solve_morph_step!(u_new, ũ, ṽ, p_new, Δt, dh, scv, mat, ch, Plv_srf, bufs; max_iter=20, tol=1e-4)
    (; K_int, r_int, K_plv, F_plv, M, K_eff, res, rhs, δu, u_trial, a_new, v_new,
       Ma, Mv, K0, Kv, F_lu, free, g_old, sdofs, ke, re, u_e,
       α_hht, γ_hht, β_hht, α_damp, β_damp) = bufs
    mfac = 1 / (β_hht * Δt^2) + (1 - α_hht) * α_damp * γ_hht / (β_hht * Δt)
    cfac = β_damp * (1 - α_hht) * γ_hht / (β_hht * Δt)   # K₀ damping → K_eff
    converged = false; iters = 0
    for iter in 1:max_iter
        iters = iter
        assemble_all!(K_int, r_int, dh, scv, u_new, mat, sdofs, ke, re, u_e)
        assemble_pressure_region!(K_plv, F_plv, dh, scv, u_new, Plv_srf, sdofs, ke, re, u_e)
        @. a_new = (u_new - ũ) / (β_hht * Δt^2)
        @. v_new = ṽ + (Δt * γ_hht) * a_new
        mul!(Ma, M, a_new); mul!(Mv, M, v_new); mul!(Kv, K0, v_new)
        @. res = Ma + (1 - α_hht) * (α_damp * Mv + β_damp * Kv + r_int - p_new * F_plv) + α_hht * g_old
        apply_zero!(res, ch)
        res_norm = norm(@views res[free])
        res_norm < tol && (converged = true; break)
        K_eff.nzval .= M.nzval .* mfac .+ cfac .* K0.nzval .+
                       (1 - α_hht) .* (K_int.nzval .- p_new .* K_plv.nzval)
        @. rhs = -res
        apply_zero!(K_eff, rhs, ch)
        lu!(F_lu, K_eff)
        ldiv!(δu, F_lu, rhs)
        α_ls = 1.0; ls_ok = false
        for _ in 1:8
            @. u_trial = u_new + α_ls * δu
            apply!(u_trial, ch)
            assemble_residual!(r_int, dh, scv, u_trial, mat, sdofs, re, u_e)
            assemble_pressure_residual!(F_plv, dh, scv, u_trial, Plv_srf, sdofs, re, u_e)
            @. a_new = (u_trial - ũ) / (β_hht * Δt^2)
            @. v_new = ṽ + (Δt * γ_hht) * a_new
            mul!(Ma, M, a_new); mul!(Mv, M, v_new); mul!(Kv, K0, v_new)
            @. res = Ma + (1 - α_hht) * (α_damp * Mv + β_damp * Kv + r_int - p_new * F_plv) + α_hht * g_old
            apply_zero!(res, ch)
            (norm(@views res[free]) ≤ res_norm) && (ls_ok = true; break)
            α_ls /= 2
        end
        u_new .= u_trial
        ls_ok || break
    end
    return converged, iters
end

# Initial state: at rest, flat reference geometry; g_old = 0 (u=v=0, p=0)
u = zeros(N_dof); apply!(u, ch)
v = zeros(N_dof)
a = zeros(N_dof)

# Compression / buckling diagnostic.  Per element, average the membrane stress
# resultant N over quadrature points, then scatter node-averaged N₁₁, N₂₂, N₁₂ and
# the minimum principal resultant N_min.  N_min < 0 ⇒ membrane compression ⇒
# wrinkling/buckling risk (the singular-tangent / Schur-collapse regime).
function membrane_resultants!(N11, N22, N12, Nmin, dh, scv, mat, u)
    fill!(N11, 0.0); fill!(N22, 0.0); fill!(N12, 0.0); fill!(Nmin, 0.0)
    cnt = zeros(Int, getnnodes(dh.grid))
    n_qp = getnquadpoints(scv)
    n_nodes_e = getnbasefunctions(scv.ip_shape)
    for cell in CellIterator(dh)
        reinit!(scv, cell)
        u_e = @views u[shelldofs(cell)]
        G₃  = scv.G₃_elem[1]
        N_avg = zero(SymmetricTensor{2,2,Float64})
        for qp in 1:n_qp
            a₁, a₂ = FerriteShells.covariant_basis(scv, qp, u_e, n_nodes_e)
            c_ms = SymmetricTensor{2,2}((dot(a₁,a₁), dot(a₁,a₂), dot(a₂,a₂)))
            Nq, _ = membrane_stress_and_tangent(mat, c_ms, scv.A_metric[qp],
                        Vec{3}(Tuple(scv.A₁[qp])), Vec{3}(Tuple(scv.A₂[qp])), G₃)
            N_avg += Nq
        end
        N_avg /= n_qp
        λ = eigvals(N_avg)   # ascending → λ[1] is the minimum principal resultant
        for nid in cell.nodes
            N11[nid] += N_avg[1,1]; N22[nid] += N_avg[2,2]; N12[nid] += N_avg[1,2]
            Nmin[nid] += λ[1];      cnt[nid]  += 1
        end
    end
    @. N11 /= max(cnt, 1); @. N22 /= max(cnt, 1)
    @. N12 /= max(cnt, 1); @. Nmin /= max(cnt, 1)
end

# Current (deformed) nodal coordinates x = X + u of the endocardium nodes only, packed
# 3×length(endo_nodes) in the local (renumbered) ordering given by `node_map`.
function endo_positions!(X, dh, u, endo_cells, node_map)
    for cell in CellIterator(dh, endo_cells)
        sd = shelldofs(cell)
        for (I, nid) in enumerate(cell.nodes)
            @views X[:, node_map[nid]] .= Ferrite.get_node_coordinate(dh.grid, nid) .+ u[sd[5I-4:5I-2]]
        end
    end
    X
end

pvd = paraview_collection("minilimo-coupled-dynamic-strong")
vtk_step = Ref(0)
resu = zeros(3, getnnodes(dh.grid))
resθ = zeros(2, getnnodes(dh.grid))
N11 = zeros(getnnodes(dh.grid)); N22 = similar(N11); N12 = similar(N11); Nmin = similar(N11)
d, G3 = director_field(dh, scv, u)
membrane_resultants!(N11, N22, N12, Nmin, dh, scv, mat, u)
VTKGridFile("minilimo-coupled-dynamic-strong-0", dh) do vtk
    write_solution(vtk, dh, u)
    Ferrite.write_node_data(vtk, resu, "ru")
    Ferrite.write_node_data(vtk, resθ, "rθ")
    Ferrite.write_node_data(vtk, d,  "director")
    Ferrite.write_node_data(vtk, G3, "G3")
    Ferrite.write_node_data(vtk, N11,  "N11")
    Ferrite.write_node_data(vtk, N22,  "N22")
    Ferrite.write_node_data(vtk, N12,  "N12")
    Ferrite.write_node_data(vtk, Nmin, "Nmin")
    for ID in 1:3; color(vtk, grid, "SRF_$ID"); end
    pvd[0.0] = vtk
end

# println("PHASE 1 — dynamic HHT-α morph")
# @printf("%-6s  %-8s  %-8s  %-8s  %-6s  %-10s\n", "step", "t [s]", "λ", "p [mmHg]", "iters", "Δt")

un = zeros(N_dof)
let t = 0.0; step = 0; Δt_cur = Δt; p = 0.0
@time while t < T_sim - 1e-10
    t_new = min(t + Δt_cur, T_sim)
    p_new = p_max * ramp(t_new)

    @. ũ = u + Δt_cur * v + (Δt_cur^2 * (0.5 - β_hht)) * a
    @. ṽ = v + (Δt_cur * (1 - γ_hht)) * a

    u_new .= ũ
    Ferrite.update!(ch, t_new)
    apply!(u_new, ch)

    converged, iters = solve_morph_step!(u_new, ũ, ṽ, p_new, Δt_cur, dh, scv, mat, ch, Plv_srf, bufs_morph;
                                         max_iter=max_iter, tol=tol)

    if converged
        step += 1
        @. a = (u_new - ũ) / (β_hht * Δt_cur^2)
        @. v = ṽ + (Δt_cur * γ_hht) * a
        mul!(Mv, M, v); mul!(Kv, K0, v)
        @. g_old = α_damp * Mv + β_damp * Kv + r_int - p_new * F_plv
        p = p_new; u .= u_new; t = t_new
        Δt_cur = min(Δt_cur * 1.2, Δt_max)
        if step % 4 == 0
            vtk_step[] += 1
            for cell in CellIterator(dh)
                sd = shelldofs(cell)
                for (I, nid) in enumerate(cell.nodes)
                    resu[:, nid] .= res[sd[5I-4:5I-2]]
                    resθ[:, nid] .= res[sd[5I-1:5I  ]]
                end
            end
            d, G3 = director_field(dh, scv, u)
            membrane_resultants!(N11, N22, N12, Nmin, dh, scv, mat, u)
            VTKGridFile("minilimo-coupled-dynamic-strong-$(vtk_step[])", dh) do vtk
                write_solution(vtk, dh, u)
                Ferrite.write_node_data(vtk, resu, "ru")
                Ferrite.write_node_data(vtk, resθ, "rθ")
                Ferrite.write_node_data(vtk, d,  "director")
                Ferrite.write_node_data(vtk, G3, "G3")
                Ferrite.write_node_data(vtk, N11,  "N11")
                Ferrite.write_node_data(vtk, N22,  "N22")
                Ferrite.write_node_data(vtk, N12,  "N12")
                Ferrite.write_node_data(vtk, Nmin, "Nmin")
                for ID in 1:3; color(vtk, grid, "SRF_$ID"); end
                pvd[t] = vtk
            end
            @printf("%-6d  %-8.3f  %-8.4f  %-8.4f  %-6d  %-10.4e\n", step, t, ramp(t), p * Pa2mmHg, iters, Δt_cur)
        end
    else
        Δt_cur /= 2
        Δt_cur < Δt_min && error("minimum Δt reached at t=$(round(t, digits=4)) s")
    end
end
    un .= u
end

# using JLD2
# jldsave("limo_dynamic_coupled_2_u0.jld2"; u=un)
# # reload if done already
# un .= load("limo_dynamic_coupled_u0.jld2")["u"]
# un .= load("limo_dynamic_coupled_2_u0.jld2")["u"]

# Freeze the fully-morphed edge configuration (t·5 ≥ T_morph → ramp = 1) for the coupled
# phase; the Dirichlet morph is held constant from here on (u, v, a carried forward).
Ferrite.update!(ch, T_sim)
u .= un
apply!(u, ch)

# actuation waveform (normalized to [0,1])
ϕᵢ(t; tC=0.10, tR=0.25, TC=0.15, TR=0.45) =
    0.0 <= (t-tC)%1 <= TC ? 0.5*(1 - cos(π*((t-tC)%1)/TC)) :
    (0.0 <= (t-tR)%1 <= TR ? 0.5*(1 + cos(π*((t-tR)%1)/TR)) : 0.0)

# Windkessel parameters, SI (Pa, m³, s) — Plv, Pa, Pv all carried in Pa.
Ra = 8.0e6    # aortic resistance   [Pa·s/m³]
Rp = 1.0e8    # peripheral resist.  [Pa·s/m³]
Rv = 5.0e5    # mitral resistance   [Pa·s/m³]
Ca = 8.0e-9   # arterial compliance [m³/Pa]
Cv = 5.0e-8   # venous compliance   [m³/Pa]
wk = (; Ra, Rp, Rv, Ca, Cv, Pscale = p_max)

# coupling controls
# RELATIVE convergence tolerance on the structural residual (was absolute).  An absolute
# force-norm tolerance is not portable between the phases: Phase-1 loads are Plv≈800 Pa,
# while Phase 2 adds Pact up to 80 kPa — order 640 N of pressure force — so the same
# absolute number is ~10² tighter exactly where the problem is hardest.  Observed
# symptom: Newton parking at |res|≈3.4e-2 N (≈5e-5 RELATIVE, converged by any practical
# standard) with a positive-definite tangent, burning all 50 iterations and collapsing Δt.
# 1e-5 relative ≈ 6e-3 N at peak actuation and ≈1e-4 N at the lightly-loaded start, i.e.
# it matches the old absolute tolerance where that one was reasonable and loosens it
# where it was not.  The 0D residuals keep their own scalings (scaleV / scaleP).
# NOTE: Phase 1 deliberately keeps its ABSOLUTE `tol` — it works, and its loads are small.
tol_cpl  = 1e-5      # relative part
tol_abs_cpl = 1e-4   # absolute floor [N] — the old tolerance, which was right at low load
max_iter = 50
dt_cpl   = 0.001   # doubles as the HHT-α time step in the coupled phase
# VTK write interval in Phase-2 time [s].  Phase 2 previously wrote a file EVERY converged
# step, with no `step % N` guard (Phase 1 has one).  With adaptive Δt collapsing to ~1e-4
# that is ~40k files and tens of GB for a 4 s beat, and it dominates both runtime and
# allocations — the expensive `director_field` / `membrane_resultants!` calls live inside
# that block.  Cadence is on TIME, not step count: Δt is adaptive, so `step % N` samples
# time non-uniformly, densest exactly where Δt collapsed.  Scalar traces (vols/pres/...)
# are still pushed every step — they are cheap and they are the actual result.
dt_vtk   = 0.01
T_beat   = 4.0    # total coupled duration [s]
T_cycle  = 1.0    # actuation period (ϕᵢ is 1-periodic) [s]
t_save   = T_beat - T_cycle   # only geometry from the last cycle is stored
# β continuation schedule: cosine walk from β_start down to β_end over the first beat.
# Zero slope at both ends, so the material never jumps — neither at the Phase-1→Phase-2
# boundary (β_cont(0) = β_start, matching Phase 1) nor on arrival at β_end.
# β is held FIXED within a time step: it is recomputed from t_new once per step, so the
# tangent always matches the residual it is linearising.
#
# With tension field off, β is now the ONLY thing standing between you and the fold, so
# read the run this way: watch `Nmin` in the VTK (which is a real compression measure again
# — relaxation used to force it to ≈0) and the β column together.  If the fold appears at
# some β* > 1, that is the honest stabilisation level for this mesh and geometry: set
# β_end = β* and report it as an artificial bending stiffener, or widen the skirt
# (x_act 0.044→0.040, y_hi 0.102→0.098) and retry at β_end = 1.
T_cont   = T_cycle   # β continuation window [s]; T_cont ≤ t_save keeps the saved cycle at β_end
β_cont(t) = t ≥ T_cont ? β_end : β_end + (β_start - β_end) * 0.5 * (1 + cos(π * t / T_cont))
@assert T_cont ≤ t_save "β must reach β_end before the saved cycle starts (T_cont=$T_cont, t_save=$t_save)"
@assert β_end ≥ 1.0 "β_end < 1 softens bending below physical — not a stabilisation"

# storages
vols = Float64[]; pres = Float64[]; pact = Float64[]
paos = Float64[]; pvns = Float64[]; tsav = Float64[]
# Endocardium (the surface `compute_volume` integrates over): cells, their nodes renumbered
# 1:n_endo, and the local connectivity.  Fixed for the run, so gathered once.
endo_cells = sort!(collect(Plv_srf))
endo_nodes = sort!(unique!(reduce(vcat, collect(Ferrite.getcells(grid, c).nodes) for c in endo_cells)))
node_map   = Dict(nid => i for (i, nid) in enumerate(endo_nodes))
conn = reduce(hcat, [node_map[n] for n in Ferrite.getcells(grid, c).nodes] for c in endo_cells)
posn = zeros(3, length(endo_nodes))   # scratch for endo_positions!
poss = Matrix{Float64}[]              # one 3×n_endo snapshot per converged step, last cycle only
tpos = Float64[]                      # sample times matching `poss`

bufs_cpl = (; K_int, r_int, K_plv, F_plv, K_pact, F_pact, K_plvpact, F_plvpact, M, K_eff,
              res, rhs, δu, v1, v2, dVdu, a_new, v_new, Ma, Mv, K0, Kv, F_lu, free, g_old,
              sdofs, ke, re, u_e, trn_dofs, rot_dofs,
              α_hht, γ_hht, β_hht, α_damp, β_damp)

# Monolithic strong-coupling step: solve the HHT-α structure AND the implicit-Euler 0D
# Windkessel simultaneously for (u_new, Plv, Pa, Pv).  `u_new` updated in place; the LV
# chamber volume is V_LV(u) = 2·V₃D(u), Plv is the coupling multiplier.  V_LVₙ/Paₙ/Pvₙ are
# the previous converged 0D state.  Returns (Plv, Pa, Pv, iters, converged, V₃D).
function solve_coupled_strong_step!(u_new, ũ, ṽ, Plv, Pa, Pv, V_LVₙ, Paₙ, Pvₙ,
                                    Pact, Δt, dh, scv, mat, ch,
                                    Plv_srf, Pact_srf, PlvPact_srf, wk, bufs;
                                    max_iter=20, tol=1e-4, tol_abs=1e-4, verbose=false,
                                    δu_cap=1.0e-3, δθ_cap=0.1)
    (; Ra, Rp, Rv, Ca, Cv, Pscale) = wk
    (; K_int, r_int, K_plv, F_plv, K_pact, F_pact, K_plvpact, F_plvpact, M, K_eff,
       res, rhs, δu, v1, v2, dVdu, a_new, v_new, Ma, Mv, K0, Kv, F_lu, free, g_old,
       sdofs, ke, re, u_e, trn_dofs, rot_dofs, α_hht, γ_hht, β_hht, α_damp, β_damp) = bufs
    mfac = 1 / (β_hht * Δt^2) + (1 - α_hht) * α_damp * γ_hht / (β_hht * Δt)
    cfac = β_damp * (1 - α_hht) * γ_hht / (β_hht * Δt)   # K₀ damping → K_eff
    converged = false; n_iter = 0; V₃D = 0.0
    scaleV = max(abs(V_LVₙ), 1e-12) / Δt
    scaleP = Pscale / Δt
    r_scale = 1.0   # structural residual scale [N]; set at iteration 1, held for the step
    for iter in 1:max_iter
        n_iter = iter
        assemble_all!(K_int, r_int, dh, scv, u_new, mat, sdofs, ke, re, u_e)
        assemble_pressure_region!(K_plv,     F_plv,     dh, scv, u_new, Plv_srf,     sdofs, ke, re, u_e)
        assemble_pressure_region!(K_pact,    F_pact,    dh, scv, u_new, Pact_srf,    sdofs, ke, re, u_e)
        assemble_pressure_region!(K_plvpact, F_plvpact, dh, scv, u_new, PlvPact_srf, sdofs, ke, re, u_e)
        @. a_new = (u_new - ũ) / (β_hht * Δt^2)
        @. v_new = ṽ + (Δt * γ_hht) * a_new
        mul!(Ma, M, a_new); mul!(Mv, M, v_new); mul!(Kv, K0, v_new)
        @. res = Ma + (1 - α_hht) * (α_damp * Mv + β_damp * Kv + r_int -
                 (Plv * F_plv + Pact * F_pact - Pact * F_plvpact)) + α_hht * g_old
        apply_zero!(res, ch)
        # chamber volume (full LV) and its gradient; dVdu = ∂compute_volume/∂u = −∂V₃D/∂u
        V₃D  = -compute_volume(dh, scv, u_new; cellset=Plv_srf)
        V_LV = 2 * V₃D
        volume_gradient!(dVdu, dh, scv, u_new; cellset=Plv_srf)
        dVdu[ch.prescribed_dofs] .= 0.0
        # 0D residuals (implicit Euler); flows on the active valve branch
        Qmv, dQmv_dPlv, dQmv_dPv = mitral_flow(Plv, Pv, Rv)
        Qao, dQao_dPlv, dQao_dPa = aortic_flow(Plv, Pa, Ra)
        R_vol = (V_LV - V_LVₙ) / Δt - (Qmv - Qao)
        R_art = (Pa - Paₙ) / Δt - (Qao / Ca + (Pv - Pa) / (Rp * Ca))
        R_ven = (Pv - Pvₙ) / Δt - ((Pa - Pv) / (Rp * Cv) - Qmv / Cv)
        res_norm = norm(@views res[free])
        # Residual scale for the RELATIVE structural tolerance, frozen at iteration 1 so
        # the target cannot drift while Newton runs.  External pressure load is the natural
        # scale; `r_int` is the fallback for the lightly-loaded start of the beat, where the
        # morph prestress — not the pressure — sets the force level.  Both restricted to the
        # free dofs, so constraint reactions do not inflate it.
        if iter == 1
            f_ref = sqrt(sum(i -> (Plv*F_plv[i] + Pact*F_pact[i] - Pact*F_plvpact[i])^2, free))
            f_int = sqrt(sum(i -> r_int[i]^2, free))
            r_scale = max(f_ref, f_int, 1e-12)
        end
        ok_c = abs(R_vol)/scaleV < tol && abs(R_art)/scaleP < tol && abs(R_ven)/scaleP < tol
        verbose && @printf("    it %2d | |res|=%.2e (rel %.2e) Rv=%+.2e Ra=%+.2e Rν=%+.2e\n",
                           iter, res_norm, res_norm/r_scale, R_vol, R_art, R_ven)
        # MIXED absolute + relative criterion: `tol·r_scale + tol_abs`.  Neither half works
        # alone here.  Purely absolute is ~10² too tight at peak actuation (r_scale ≈ 640 N),
        # which was the original defect.  But purely relative is far too tight at LOW load —
        # measured r_scale ≈ 0.012 N early in the beat, where tol·r_scale demands 1.2e-7 N
        # against the old 1e-4 N, three orders tighter, making easy steps work for nothing.
        # The sum degrades to the absolute floor when the structure is barely loaded and to
        # the relative term when it is heavily loaded, which is the behaviour we want in both.
        (res_norm < tol * r_scale + tol_abs && ok_c && iter != 1) && (converged = true; n_iter = iter - 1; break)
        # structural tangent + two back-substitutions (shared with the volume row)
        K_eff.nzval .= M.nzval .* mfac .+ cfac .* K0.nzval .+ (1 - α_hht) .* (K_int.nzval .-
                       Plv .* K_plv.nzval .- Pact .* K_pact.nzval .+ Pact .* K_plvpact.nzval)
        @. rhs = -res
        apply_zero!(K_eff, rhs, ch)
        lu!(F_lu, K_eff)
        ldiv!(v1, F_lu, rhs)
        ldiv!(v2, F_lu, F_plv); v2 .*= (1 - α_hht)   # ∂R_s/∂Plv = −(1−α)·F_plv → δu = v1 + δPlv·v2
        # aᵀ = (1/Δt)·∂V_LV/∂u = −(2/Δt)·dVdu  (volume-row coupling to δu)
        aTv1 = -(2 / Δt) * dot(dVdu, v1)
        aTv2 = -(2 / Δt) * dot(dVdu, v2)
        # TANGENT-HEALTH DIAGNOSTICS — zero extra cost, both quantities already formed.
        # qN = rhsᵀK_eff⁻¹rhs and qP = (1−α)·F_plvᵀK_eff⁻¹F_plv are positive iff K_eff is
        # positive definite along the Newton / pressure directions.  qN < 0 ⇒ the Newton
        # step is an ASCENT direction ⇒ the shell has buckled.  aTv2 is the Schur coupling
        # entering j11; as it collapses toward zero the 3×3 0D Jacobian goes singular —
        # a DIFFERENT failure from an indefinite K_eff, and worth telling apart before
        # chasing the wrong one (cf. the S_min/δp diagnostics in the static-weak script).
        qN = dot(v1, rhs); qP = dot(v2, F_plv)
        verbose && @printf("       aTv2=%+.3e  qN=%+.3e  qP=%+.3e%s\n",
                           aTv2, qN, qP, (qN < 0 || qP < 0) ? "   <-- K_eff INDEFINITE" : "")
        # 3×3 condensed 0D Jacobian J·(δPlv,δPa,δPv) = rhs_c  (column-major Tensor)
        j11 = (-dQmv_dPlv + dQao_dPlv) + aTv2
        j12 = dQao_dPa
        j13 = -dQmv_dPv
        j21 = -dQao_dPlv / Ca
        j22 = 1/Δt - dQao_dPa / Ca + 1/(Rp*Ca)
        j23 = -1/(Rp*Ca)
        j31 = dQmv_dPlv / Cv
        j32 = -1/(Rp*Cv)
        j33 = 1/Δt + 1/(Rp*Cv) + dQmv_dPv / Cv
        Jt  = Tensor{2,3}((j11, j21, j31, j12, j22, j32, j13, j23, j33))
        bt  = Vec{3}((-R_vol - aTv1, -R_art, -R_ven))
        δc  = inv(Jt) ⋅ bt
        δPlv, δPa, δPv = δc[1], δc[2], δc[3]
        # TRUST-REGION STEP CAP.  On the wrinkled branch the tension-field membrane
        # tangent is genuinely indefinite (one eigenvalue ≈ −2.4% of the largest, and
        # non-symmetric — FD-verified), so K_eff can be indefinite and this Newton
        # direction can be an ASCENT direction.  `solve_morph_step!` absorbs that with
        # backtracking; this monolithic step has NO line search, so without a cap the
        # bad step is taken at full length and the fold runs away in one iteration.
        # Scaling the WHOLE Newton direction (u and the three pressures together) keeps
        # the step consistent.  Set δu_cap = δθ_cap = Inf to recover the uncapped solver.
        @. δu = v1 + δPlv * v2
        s_n = min(1.0,
                  δu_cap / max(maximum(abs, @view δu[trn_dofs]), eps()),
                  δθ_cap / max(maximum(abs, @view δu[rot_dofs]), eps()))
        @. u_new = u_new + s_n * δu
        Plv += s_n * δPlv; Pa += s_n * δPa; Pv += s_n * δPv
        apply!(u_new, ch)
    end
    return Plv, Pa, Pv, n_iter, converged, V₃D
end

# initial (full-LV) cavity volume from the morphed state
V_LV0 = -2 * compute_volume(dh, scv, u; cellset=Plv_srf)   # m³
println("Initial volume of the device: ", round(V_LV0 * m3_to_ml; digits=4), " ml")

println("\nPHASE 2 — monolithic strong 3D-0D coupling (adaptive Δt, Δt₀=$(dt_cpl) s)")
println("      t [s] |  p [mmHg]   |  Vlv_full [ml]  |  Pact [mmHg]  | iters |    Δt [s]  |    β")

# initial 0D state [Pa]: filled ventricle (Plv=Pv=p_max), arterial at 80 mmHg
Pa0 = 80.0 / Pa2mmHg
Pv0 = p_max
# Adaptive Δt driven by 3D-0D coupling convergence (same policy as the Phase-1 morph):
# a converged monolithic step commits and grows Δt by 1.2× (capped at Δt_max); a failed
# step is discarded (structural + 0D history untouched), Δt halved, and the step retried.
@time let V_LVₙ = V_LV0, Paₙ = Pa0, Pvₙ = Pv0, t_cpl = 0.0, Plv = p_max, Pa = Pa0, Pv = Pv0
    step = 0
    Δt_cur = dt_cpl
    t_next_vtk = 0.0   # next Phase-2 time at which a VTK frame is written
    while t_cpl < T_beat - Δt_min
        t_new  = min(t_cpl + Δt_cur, T_beat)
        Δt_cur = t_new - t_cpl   # clip the final step to land exactly on T_beat

        Pact_mmHg = 600 * ϕᵢ(t_new; tC=0.1, tR=0.4, TC=0.3, TR=0.3)
        Pact = Pact_mmHg / Pa2mmHg

        # Bending-scale continuation: one β per time step, held fixed through the Newton
        # solve.  A failed step that halves Δt lands on a new t_new and simply re-evaluates
        # β there — no state to roll back.  Rebuilding the (immutable, isbits) material is
        # a few ns; the assembly cost is unchanged.
        β_cur   = β_cont(t_new)
        mat_cur = mat_β(β_cur)

        # HHT-α predictors (Δt = Δt_cur), morph BC frozen.
        @. ũ = u + Δt_cur * v + (Δt_cur^2 * (0.5 - β_hht)) * a
        @. ṽ = v + (Δt_cur * (1 - γ_hht)) * a
        u_new .= ũ
        apply!(u_new, ch)

        # Trial pressures start from the last committed 0D state; only adopted on success.
        Plv_t, Pa_t, Pv_t, n_iter, converged, V₃D = solve_coupled_strong_step!(
            u_new, ũ, ṽ, Plv, Pa, Pv, V_LVₙ, Paₙ, Pvₙ, Pact, Δt_cur, dh, scv, mat_cur, ch,
            Plv_srf, Pact_srf, PlvPact_srf, wk, bufs_cpl; max_iter=max_iter, tol=tol_cpl, tol_abs=tol_abs_cpl, verbose=false)

        if !converged
            Δt_cur /= 2
            Δt_cur < Δt_min && error("minimum Δt reached at t=$(round(t_cpl, digits=4)) s")
            continue   # discard trial state (u, v, a, 0D history all unchanged), retry
        end

        step += 1
        t_cpl = t_new
        Plv, Pa, Pv = Plv_t, Pa_t, Pv_t

        # commit dynamic structural state + advance 0D history
        @. a = (u_new - ũ) / (β_hht * Δt_cur^2)
        @. v = ṽ + (Δt_cur * γ_hht) * a
        mul!(Mv, M, v); mul!(Kv, K0, v)
        @. g_old = α_damp * Mv + β_damp * Kv + r_int - (Plv * F_plv + Pact * F_pact - Pact * F_plvpact)
        u .= u_new
        V_LVₙ = 2 * V₃D; Paₙ = Pa; Pvₙ = Pv

        # VTK on a TIME cadence (dt_vtk), not every step.  director_field /
        # membrane_resultants! live in here, so this guard throttles the dominant
        # per-step allocation as well as the file count.
        if t_cpl ≥ t_next_vtk - 1e-12
            vtk_step[] += 1
            for cell in CellIterator(dh)
                sd = shelldofs(cell)
                for (I, nid) in enumerate(cell.nodes)
                    resu[:, nid] .= res[sd[5I-4:5I-2]]
                    resθ[:, nid] .= res[sd[5I-1:5I  ]]
                end
            end
            d, G3 = director_field(dh, scv, u)
            membrane_resultants!(N11, N22, N12, Nmin, dh, scv, mat_cur, u)
            VTKGridFile("minilimo-coupled-dynamic-strong-$(vtk_step[])", dh) do vtk
                write_solution(vtk, dh, u)
                Ferrite.write_node_data(vtk, resu, "ru")
                Ferrite.write_node_data(vtk, resθ, "rθ")
                Ferrite.write_node_data(vtk, d,  "director")
                Ferrite.write_node_data(vtk, G3, "G3")
                Ferrite.write_node_data(vtk, N11,  "N11")
                Ferrite.write_node_data(vtk, N22,  "N22")
                Ferrite.write_node_data(vtk, N12,  "N12")
                Ferrite.write_node_data(vtk, Nmin, "Nmin")
                for ID in 1:3; color(vtk, grid, "SRF_$ID"); end
                pvd[T_sim + t_cpl] = vtk
            end
            # Advance past t_cpl even if a single Δt jumped several intervals.
            while t_next_vtk ≤ t_cpl; t_next_vtk += dt_vtk; end
        end
        @printf("  %9.4f | %11.4f | %14.4f | %14.4f | %5d | %.3e | %6.3f\n",
                t_cpl, Plv * Pa2mmHg, 2V₃D * m3_to_ml, Pact_mmHg, n_iter, Δt_cur, β_cur)

        push!(tsav, t_cpl)             # actual (non-uniform) sample time [s]
        push!(vols, 2V₃D * m3_to_ml)   # full LV volume [ml]
        push!(pres, Plv * Pa2mmHg)     # LV pressure [mmHg]
        push!(pact, Pact_mmHg)
        push!(paos, Pa * Pa2mmHg)
        push!(pvns, Pv * Pa2mmHg)
        if t_cpl ≥ t_save   # geometry of the last cycle only
            push!(tpos, t_cpl)
            push!(poss, copy(endo_positions!(posn, dh, u, endo_cells, node_map)))   # [m]
        end

        Δt_cur = min(Δt_cur * 1.2, Δt_max)   # grow after a converged step
    end
end
close(pvd)

jldsave("minilimo-coupled-dynamic-strong-positions.jld2";
        positions=poss, connectivity=conn, t=tpos, t_all=tsav,
        vols=vols, pres=pres, pact=pact, paos=paos, pvns=pvns)

using Plots
times = tsav   # adaptive Δt → non-uniform sample times
p1 = plot(times, [vols, pres, paos, pvns], xlabel="Time [s]",
          label=["Vlv" "Plv" "Pao" "Pv"], lw=2, legend=:right)
p2 = plot(vols, pres, label=:none, xlim=extrema(vols).+(-10,10), ylims=(0, 100),
          xlabel="Volume [ml]", ylabel="Pressure [mmHg]", lw=2, linez=round.(times, RoundUp))
plot(p1, p2)
# savefig("minilimo-coupled-dynamic-strong-N$Np.png")