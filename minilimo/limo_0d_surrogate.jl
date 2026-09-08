# ══════════════════════════════════════════════════════════════════════════════════════
#  0D SURROGATE OF THE miniLIMO 3D-0D COUPLED BEAT
# ══════════════════════════════════════════════════════════════════════════════════════
#
# WHAT THIS IS.  A closed-loop 0D model that reproduces the hemodynamics of
# `limo_coupled_dynamic_strong*.jl` in about a second instead of a full coupled run.  The
# 3D shell is replaced by a two-parameter constitutive law fitted to a completed coupled
# run; the Windkessel, the valves and the actuation waveform are IDENTICAL to the coupled
# file.  Use it to choose Windkessel coefficients, actuation amplitude/timing and the
# Phase-2 initial state BEFORE committing to a coupled run.
#
# HOW TO RUN
#     julia --project=examples minilimo/limo_0d_surrogate.jl        # demo + calibration
#     julia --project=examples -e 'include("minilimo/limo_0d_surrogate.jl"); ...'
# (the `examples` environment is the one carrying JLD2/Plots for the minilimo scripts)
#
# ─── THE DEVICE LAW ───────────────────────────────────────────────────────────────────
# Fitting Plv = α·Pact + β·V + γ by least squares over the last cycle of a coupled run
# gives, for the 750 mmHg / new-mesh configuration's 600 mmHg predecessor:
#
#     Plv = 0.17157·Pact + 0.36273·V − 80.245        [mmHg, mL]
#     R² = 0.933,  RMSE = 9.77 mmHg
#
# Read physically, the device is a PRESSURE SOURCE of α·Pact in series with a FIXED
# chamber elastance β and an unstressed volume −γ/β:
#
#   α = 0.172        actuator-to-chamber transmission.  Caps the generated pressure at
#                    α·Pact_max — 103 mmHg at Pact=600, 129 mmHg at Pact=750.  No choice
#                    of Windkessel coefficients produces a systolic pressure above this.
#   β = 0.363 mmHg/mL   chamber elastance, CONSTANT.  The device has no systolic
#                    stiffening, so afterload sensitivity is dSV/dPa = −1/β = −2.76 mL per
#                    mmHg — about 7x a real LV (E_es ≈ 2.5 → −0.4 mL/mmHg).  This is why
#                    raising Rp to lift MAP costs so much stroke volume.
#   −γ/β = 221.2 mL  unstressed volume.  Below it, with the actuator off, Plv is NEGATIVE.
#                    Ejecting a physiological SV from EDV ≈ 260 mL puts end-systole ~34 mL
#                    under, so keeping Plv positive through relaxation is a race between
#                    the relaxation rate TR and the filling rate 1/Rv.  See `min_Plv` in
#                    the results and the calibration constraint below.
#
# ─── LIMITATIONS (read before trusting a number) ──────────────────────────────────────
#  * QUASI-STATIC.  No shell inertia, no Rayleigh damping, no HHT-α.  The coupled file has
#    all three.  Transients on the ~10 ms scale are not represented.
#  * LINEAR in (Pact, V).  Wrinkling, follower-pressure softening and large-rotation
#    stiffening are all outside the model.  RMSE 9.8 mmHg against the run it was fitted to.
#  * KNOWN BIAS AT THE RELAXATION TROUGH: the surrogate read +9.4 mmHg where the coupled
#    run measured +4.6, i.e. ~4.8 mmHg OPTIMISTIC.  When using it to keep Plv positive,
#    demand a surrogate margin of at least +6 mmHg.  `calibrate` defaults to that.
#  * CANNOT PREDICT BUCKLING or Newton failure — there is no shell in here at all.  The
#    follower-softening ratio that drives the fold scales with Pact and is the coupled
#    file's problem, not this one's.
#  * EXTRAPOLATION.  Raising Pact past the amplitude the law was fitted at is a linear
#    extrapolation.  REFIT (`fit_device_law`) from the new run afterwards.
#
# Units are CLINICAL throughout (mL, mmHg, s).  `si_table` prints the Pa/m³/s values to
# paste into the coupled file.
# ══════════════════════════════════════════════════════════════════════════════════════

using Printf, LinearAlgebra

const MMHG_PER_PA  = 0.00750062          # matches Pa2mmHg in the coupled files
const ML_PER_M3    = 1.0e6
# R[Pa·s/m³] = R[mmHg·s/mL] · ML_PER_M3 / MMHG_PER_PA   (= /7.50062e-9)
# C[m³/Pa]   = C[mL/mmHg]   · MMHG_PER_PA / ML_PER_M3   (= /1.33322e8)
# NOTE the two run OPPOSITE ways: resistance is pressure/flow, compliance is volume/
# pressure, so the same pair of factors appears inverted.  Easy to get backwards.
R_to_SI(R)   = R * ML_PER_M3 / MMHG_PER_PA   # mmHg·s/mL → Pa·s/m³
R_to_clin(R) = R * MMHG_PER_PA / ML_PER_M3   # Pa·s/m³   → mmHg·s/mL
C_to_SI(C)   = C * MMHG_PER_PA / ML_PER_M3   # mL/mmHg   → m³/Pa
C_to_clin(C) = C * ML_PER_M3 / MMHG_PER_PA   # m³/Pa     → mL/mmHg

_mean(x) = sum(x) / length(x)

# ─── device constitutive law ──────────────────────────────────────────────────────────
struct DeviceLaw{T}
    α::T        # actuator transmission [-]
    β::T        # chamber elastance [mmHg/mL]
    γ::T        # offset [mmHg]
end
(d::DeviceLaw)(Pact, V) = d.α * Pact + d.β * V + d.γ
unstressed_volume(d::DeviceLaw) = -d.γ / d.β
afterload_sensitivity(d::DeviceLaw) = -1 / d.β          # mL of SV per mmHg of afterload
pressure_ceiling(d::DeviceLaw, Pact_max) = d.α * Pact_max

# Fitted from minilimo-coupled-dynamic-strong-positions.jld2 (600 mmHg run, 2026-08-31).
const DEVICE_600 = DeviceLaw(0.17157, 0.36273, -80.245)

"""
    fit_device_law(path; t_from=3.0)

Least-squares fit of `Plv = α·Pact + β·V + γ` over the samples with `t_all ≥ t_from` in a
`*-positions.jld2` written by a coupled run.  Returns `(law, R², RMSE)`.  REFIT AFTER ANY
RUN AT A NEW ACTUATION AMPLITUDE — the law is linear and does not extrapolate for free.
"""
function fit_device_law(path::AbstractString; t_from = 3.0)
    JLD2 = Base.require(Base.PkgId(Base.UUID("033835bb-8acc-5ee8-8aae-3f567f8a3819"), "JLD2"))
    d = JLD2.load(path)
    m = d["t_all"] .≥ t_from
    P, V, Pact = d["pres"][m], d["vols"][m], d["pact"][m]
    A = hcat(Pact, V, ones(length(V)))
    c = A \ P
    pred = A * c
    R2   = 1 - sum(abs2, P .- pred) / sum(abs2, P .- _mean(P))
    RMSE = sqrt(_mean(abs2.(P .- pred)))
    DeviceLaw(c[1], c[2], c[3]), R2, RMSE
end

# ─── actuation waveform — VERBATIM from the coupled files, do not diverge ─────────────
ϕᵢ(t; tC=0.10, tR=0.25, TC=0.15, TR=0.45) =
     0.0 <= (t-tC)%1 <= TC ? 0.5*(1 - cos(π*((t-tC)%1)/TC)) :
    (0.0 <= (t-tR)%1 <= TR ? 0.5*(1 + cos(π*((t-tR)%1)/TR)) :
    (TC <= (t-tC)%1 <= TR ? 1.0 : 0.0))

# ─── default parameter set: matches limo_coupled_dynamic_strong_physio.jl ─────────────
const WK_PHYSIO = (Ra=0.030, Rp=1.10, Rv=0.0075, Ca=1.60, Cv=60.0, Rc=75006.0)
const ACT_PHYSIO = (Pact_max=750.0, tC=0.10, tR=0.30, TC=0.15, TR=0.30)
const IC_PHYSIO  = (V0=250.0, Pa0=90.0, Pv0=14.0)
# The uncalibrated original, for A/B comparison.
const WK_ORIG  = (Ra=0.060, Rp=0.750, Rv=0.00375, Ca=1.0666, Cv=6.666, Rc=75.0)
const ACT_ORIG = (Pact_max=600.0, tC=0.10, tR=0.35, TC=0.15, TR=0.35)
const IC_ORIG  = (V0=257.0, Pa0=80.0, Pv0=6.0)

"""
    run0d(dev=DEVICE_600; Ra, Rp, Rv, Ca, Cv, Rc, Pact_max, tC, tR, TC, TR,
                          V0, Pa0, Pv0, nbeat=25, dt=2e-5, keep_traces=false)

Integrate the closed loop to a limit cycle and return metrics from the LAST beat.

Topology, identical to the coupled file:
    LV --Ra--> [Ca, Pa] --Rp--> [Cv, Pv] --Rv--> LV
Valves are resistive diodes (`Rc` when closed).  The loop conserves stressed volume
exactly — V_tot = V_LV + Ca·Pa + Cv·Pv is fixed by the initial condition, so PRELOAD IS
SET BY (V0, Pa0, Pv0), not by any resistance.

`Pao_sys`/`Pao_dia`/`MAP` are ROOT pressures, `Pa + Ra·Qao`, which for a resistive valve
equals `max(Plv, Pa)`.  `Pa` alone is the distal compliance-node pressure and understates
systolic by the whole Ra·Qao drop.
"""
function run0d(dev::DeviceLaw = DEVICE_600;
               Ra, Rp, Rv, Ca, Cv, Rc = 75006.0,
               Pact_max, tC = 0.10, tR = 0.30, TC = 0.15, TR = 0.30,
               V0, Pa0, Pv0, nbeat = 25, dt = 2e-5, keep_traces = false)
    V, Pa, Pv = V0, Pa0, Pv0
    n = round(Int, nbeat / dt)
    ts=Float64[]; Vs=Float64[]; Ps=Float64[]; Pas=Float64[]; Pvs=Float64[]
    Qas=Float64[]; Qms=Float64[]; Pacts=Float64[]
    for i in 1:n
        t    = (i-1) * dt
        Pact = Pact_max * ϕᵢ(t; tC=tC, tR=tR, TC=TC, TR=TR)
        P    = dev(Pact, V)
        Qao  = P  > Pa ? (P - Pa)/Ra : (P - Pa)/Rc
        Qmv  = Pv > P  ? (Pv - P)/Rv : (Pv - P)/Rc
        V  += dt * (Qmv - Qao)
        Pa += dt * (Qao - (Pa - Pv)/Rp) / Ca
        Pv += dt * ((Pa - Pv)/Rp - Qmv) / Cv
        if t ≥ nbeat - 1.0 && i % 10 == 0
            push!(ts, t); push!(Vs, V); push!(Ps, P); push!(Pas, Pa); push!(Pvs, Pv)
            push!(Qas, max(Qao, 0.0)); push!(Qms, max(Qmv, 0.0)); push!(Pacts, Pact)
        end
    end
    EDV, ESV = maximum(Vs), minimum(Vs)
    SV    = EDV - ESV
    proot = max.(Ps, Pas)
    k     = argmin(Ps)
    base = (; EDV, ESV, SV, EF = 100SV/EDV, CO = SV*60/1000,
              Plv_pk = maximum(Ps), Plv_min = minimum(Ps),
              t_min = ts[k] % 1, V_at_min = Vs[k], Pact_at_min = Pacts[k],
              frac_neg = 100 * count(<(0), Ps) / length(Ps),
              Pao_sys = maximum(proot), Pao_dia = minimum(proot),
              PP = maximum(proot) - minimum(proot), MAP = _mean(proot),
              Pv_mean = _mean(Pvs), Pv_swing = maximum(Pvs) - minimum(Pvs),
              Qao_pk = maximum(Qas), Qmv_pk = maximum(Qms),
              V_unstressed = unstressed_volume(dev),
              below_unstressed = unstressed_volume(dev) - ESV)
    keep_traces ? merge(base, (; t=ts, V=Vs, Plv=Ps, Pa=Pas, Pv=Pvs, Pao=proot,
                                 Qao=Qas, Qmv=Qms, Pact=Pacts)) : base
end

# ─── reporting ────────────────────────────────────────────────────────────────────────
const REFERENCE = """
  healthy adult:  SV 60-100 mL   CO 4.5-6.0 L/min   EF 55-70%
                  Pao 120/80 (PP ~40)   MAP ~93 mmHg   LAP 8-12 (swing 3-5)
                  peak Qao 400-600   peak Qmv 400-600 mL/s"""

function report(r; label = "0D surrogate")
    @printf("\n── %s ───────────────────────────────────────────\n", label)
    @printf("  EDV %6.1f  ESV %6.1f  SV %5.1f mL   CO %4.2f L/min   EF %4.1f%%\n",
            r.EDV, r.ESV, r.SV, r.CO, r.EF)
    @printf("  Plv %6.1f .. %6.1f mmHg | Pao(root) %5.1f/%5.1f  PP %4.1f  MAP %5.1f\n",
            r.Plv_pk, r.Plv_min, r.Pao_sys, r.Pao_dia, r.PP, r.MAP)
    @printf("  Pv %5.1f ± %4.1f mmHg | peak Qao %5.0f  Qmv %5.0f mL/s\n",
            r.Pv_mean, r.Pv_swing, r.Qao_pk, r.Qmv_pk)
    @printf("  unstressed volume %.1f mL; ESV sits %.1f mL below it\n",
            r.V_unstressed, r.below_unstressed)
    if r.Plv_min < 0
        @printf("  ⚠ Plv NEGATIVE for %.1f%% of the cycle (min at t=%.3f, V=%.1f, Pact=%.1f)\n",
                r.frac_neg, r.t_min, r.V_at_min, r.Pact_at_min)
        @printf("    relaxation is outrunning filling — lengthen TR, or lower Rv, or raise Pv0\n")
    else
        @printf("  Plv trough +%.1f mmHg at t=%.3f (surrogate reads ~4.8 HIGH → expect ~%+.1f)\n",
                r.Plv_min, r.t_min, r.Plv_min - 4.8)
    end
    return r
end

"""
    si_table(wk; Pact_max=nothing)

Print the SI values to paste into `limo_coupled_dynamic_strong*.jl`, alongside the
Regazzoni et al. (JCP 2022, Part II Tab. 4) reference column.
"""
function si_table(wk; Pact_max = nothing)
    ref = (Ra="R_min 0.0075", Rp="R_AR+R_VEN 1.06", Rv="R_min 0.0075",
           Ca="C_AR 1.2", Cv="C_VEN 60.0", Rc="R_max 75006")
    println("\n  name        SI                 clinical        Regazzoni Tab. 4")
    println("  ─────────── ────────────────── ─────────────── ──────────────────")
    for k in (:Ra, :Rp, :Rv)
        @printf("  %-11s %-18.3e %-15.4g %s\n", k, R_to_SI(wk[k]), wk[k], ref[k])
    end
    for k in (:Ca, :Cv)
        @printf("  %-11s %-18.3e %-15.4g %s\n", k, C_to_SI(wk[k]), wk[k], ref[k])
    end
    haskey(wk, :Rc) && @printf("  %-11s %-18.3e %-15.4g %s\n",
                               "R_closed", R_to_SI(wk.Rc), wk.Rc, ref.Rc)
    Pact_max === nothing || @printf("  %-11s %-18s %-15.4g mmHg\n", "Pact_max", "—", Pact_max)
end

"""
    calibrate(dev=DEVICE_600; MAP_target=93.0, SV_target=70.0, min_Plv=6.0, ...)

Grid search over the knobs that actually move the operating point, subject to a hard
constraint that the relaxation-phase trough in Plv stays above `min_Plv`.  The default
+6 mmHg is the surrogate's ~4.8 mmHg optimistic bias plus a small margin — do not lower
it without a reason.  Returns `(best_params, best_result)`.
"""
function calibrate(dev::DeviceLaw = DEVICE_600;
                   MAP_target = 93.0, SV_target = 70.0, min_Plv = 6.0,
                   Pact_range = 650.0:25.0:900.0, Rp_range = 0.85:0.05:1.60,
                   Pv0_range = (10.0, 12.0, 14.0, 16.0), TR_range = (0.25, 0.30, 0.35),
                   Rv_range  = (0.0056, 0.0075), fixed = (Ra=0.030, Ca=1.60, Cv=60.0),
                   Qmv_soft = 600.0, nbeat = 15, dt = 5e-5)
    best = nothing; bs = Inf
    for Pact in Pact_range, Rp in Rp_range, Pv0 in Pv0_range, TR in TR_range, Rv in Rv_range
        r = run0d(dev; fixed..., Rp=Rp, Rv=Rv, Pact_max=Pact, TR=TR,
                  V0=250.0, Pa0=90.0, Pv0=Pv0, nbeat=nbeat, dt=dt)
        r.Plv_min < min_Plv && continue
        s = ((r.MAP - MAP_target)/3)^2 + ((r.SV - SV_target)/3)^2 +
            ((r.Qmv_pk - Qmv_soft)/400)^2
        s < bs && (bs = s; best = (; fixed..., Rp, Rv, Pact_max=Pact, TR, Pv0))
    end
    best === nothing && error("no configuration met min_Plv = $min_Plv; widen the ranges")
    p = best
    r = run0d(dev; p.Ra, p.Ca, p.Cv, p.Rp, p.Rv, Pact_max=p.Pact_max, TR=p.TR,
              V0=250.0, Pa0=90.0, Pv0=p.Pv0, nbeat=30, dt=2e-5)
    best, r
end

# ─── demo ─────────────────────────────────────────────────────────────────────────────
if abspath(PROGRAM_FILE) == @__FILE__
    d = DEVICE_600
    @printf("device law: Plv = %.5f·Pact + %.5f·V %+.3f   [mmHg, mL]\n", d.α, d.β, d.γ)
    @printf("  unstressed volume   %.1f mL\n", unstressed_volume(d))
    @printf("  afterload sens.     %.2f mL per mmHg  (human LV ≈ -0.4)\n", afterload_sensitivity(d))
    @printf("  pressure ceiling    %.1f mmHg at Pact=600 | %.1f at Pact=750\n",
            pressure_ceiling(d, 600.0), pressure_ceiling(d, 750.0))

    report(run0d(d; WK_ORIG..., ACT_ORIG..., IC_ORIG...);   label = "ORIGINAL coefficients")
    report(run0d(d; WK_PHYSIO..., ACT_PHYSIO..., IC_PHYSIO...); label = "PHYSIO file (shipped)")
    println(REFERENCE)
    si_table(WK_PHYSIO; Pact_max = ACT_PHYSIO.Pact_max)

    println("\nre-running the constrained calibration (a few seconds)…")
    p, r = calibrate(d)
    println("  best: ", p)
    report(r; label = "calibrated")
end
