using FerriteShells, LinearAlgebra, Printf, WriteVTK
include(joinpath(@__DIR__, "util.jl"))

# Full-device (no-symmetry) dynamic strongly-coupled miniLIMO — WIP scaffold.
#
# Unlike the half-model scripts (which build `make_minilimo_grid` with symmetry
# planes), this loads the full 3D device mesh `p6_large/geom.inp` (S4/Q4 shell,
# flat z=0 reference) with its 16 `SRF_*` element sets.  The two mirror halves
# {SRF_1..8} and {SRF_9..16} together form the closed device; SRF_1/SRF_9 are the
# large endocardium patches, the rest are actuator pouches.
#
# GOAL: a monolithic HHT-α structure + implicit-Euler 0D Windkessel solve, as in
# `limo_coupled_dynamic_strong.jl`, but on the full device.
#
# THIS STEP: load + scale the mesh, set up the Q4/MITC4 shell, the opposite-z
# opening morph BC on the two base rims, the signed follower-pressure weight map,
# and PHASE 1 = the dynamic HHT-α morph (pressure-free for now; the weighted Plv/
# Pact follower assembly + 0D Windkessel coupling come next).

meshfile = "/home/marin/Workspace/HHH/code/miniLIMO/p6_large/geom.inp"
grid = get_ferrite_grid(meshfile)
n_srf = 16
# The .inp is scaled ×10 in length vs SI; rescale nodes to metres so the SI
# material/density/thickness below are consistent.
const LSCALE = 0.1
let nodes = Ferrite.getnodes(grid)
    for i in eachindex(nodes); nodes[i] = Node(LSCALE * nodes[i].x); end
end
@printf("loaded %s\n  %d cells, %d nodes, %s  (nodes scaled ×%.2f → m)\n",
        basename(meshfile), Ferrite.getncells(grid), Ferrite.getnnodes(grid), eltype(grid.cells), LSCALE)

# material (SI: Pa, kg/m³, m) — same soft TPU as the half-model coupled script.
# tension_field=true: the thin endocardium wrinkles instead of snapping at the
# base-rim corners (sheds compressive principal membrane stress).
ρ   = 1200.0
mat = LinearElastic(20e6, 0.3, 0.001; tension_field=true, ε_tf=1e-3)

ip  = Lagrange{RefQuadrilateral, 1}()
qr  = QuadratureRule{RefQuadrilateral}(2)
scv = ShellCellValues(qr, ip, ip; mitc=MITC4)

dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

# Morph BC edge: the y=0 base rim of the endocardium patches SRF_1/SRF_9 — the
# open edge of the closed pouch (both sheets are welded everywhere else).  The two
# sheets keep DISTINCT (coincident) node ids here, so the nodeset unions both and
# the morph drives both base rims identically.  Full-device analogue of the
# half-model `edge`.
# Base-rim nodes of each sheet separately (coincident coords, distinct ids).
base_nodes(k) = (s = Set{Int}(); for c in getcellset(grid, "SRF_$k"); union!(s, grid.cells[c].nodes); end;
                 ymin = minimum(get_node_coordinate(grid, n)[2] for n in s);
                 Set(n for n in s if get_node_coordinate(grid, n)[2] < ymin + 1e-6))
base1 = base_nodes(1)                    # SRF_1 base rim (folds +z)
base9 = base_nodes(9)                    # SRF_9 base rim (folds −z)
addnodeset!(grid, "base1", base1)
addnodeset!(grid, "base9", base9)
addnodeset!(grid, "base9_only", setdiff(base9, base1))   # drop the 2 shared corner ids
@printf("  morph edges: SRF_1 base %d, SRF_9 base %d (shared %d)\n",
        length(base1), length(base9), length(intersect(base1, base9)))

# BC: elliptic-arc morph on each base rim (same map_initial/Ar as the half-model),
# but the two sheets fold to OPPOSITE z so the pouch opens: SRF_1 up (zsign=+1),
# SRF_9 down (zsign=−1).  Δx → u_x (same for both, rims stay coincident in-plane),
# ellipse height → ±u_z; u_y = 0; θ = 0.  The shared corner ids (u_z=0 there) are
# constrained once via base1; base9_only excludes them to avoid a duplicate DOF.
# No symmetry planes — the morph edges alone remove the rigid-body modes.
T_morph = 2.0
T_sim   = 2.0
ramp(t) = t < T_morph ? 0.5 * (1 - cos(π * t / T_morph)) : 1.0
morph_up   = generate_boundary_function(grid, "base1"; ramp=ramp, zsign=+1)
morph_down = generate_boundary_function(grid, "base9"; ramp=ramp, zsign=-1)

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getnodeset(grid, "base1"),      (x, t) -> morph_up(x, t),   [1, 3]))
add!(ch, Dirichlet(:u, getnodeset(grid, "base9_only"), (x, t) -> morph_down(x, t), [1, 3]))
add!(ch, Dirichlet(:u, getnodeset(grid, "base1"),      x -> 0.0, [2]))
add!(ch, Dirichlet(:u, getnodeset(grid, "base9_only"), x -> 0.0, [2]))
add!(ch, Dirichlet(:θ, getnodeset(grid, "base1"),      x -> zeros(2), [1, 2]))
add!(ch, Dirichlet(:θ, getnodeset(grid, "base9_only"), x -> zeros(2), [1, 2]))
close!(ch); Ferrite.update!(ch, 0.0)     # start flat; the morph ramps over Phase 1
edge_nodes = union(base1, base9)         # for the VTK edge marker

# Follower-pressure surfaces — SAME structure as the half-model coupled scripts,
#   F_ext = Plv·F_plv + Pact·F_pact − Pact·F_plvpact,
# on the full closed device, so a per-cell psign (+1 upper half {SRF_1..8}, −1
# lower half {SRF_9..16}) flips the mirror half (as in limo_dynamic_full.jl):
#   endocardium  SRF_1/9         : Plv        (F_plv)
#   actuator ext SRF_2,3,4/10-12 : Pact       (F_pact)
#   footprint    SRF_6,7,8/14-16 : Plv − Pact (F_plv + −F_plvpact, double layer)
#   SRF_5/13 unloaded
srfset(ks) = reduce(∪, (getcellset(grid, "SRF_$k") for k in ks))
endo_srf = srfset((1, 9))
ext_srf  = srfset((2, 3, 4, 10, 11, 12))
foot_srf = srfset((6, 7, 8, 14, 15, 16))
Plv_srf     = endo_srf ∪ foot_srf     # Plv on endocardium + footprint
Pact_srf    = ext_srf                 # +Pact on actuator exteriors
PlvPact_srf = foot_srf                # −Pact on footprints (opposes Plv)
psign = ones(Ferrite.getncells(grid))
for k in 9:16, cid in getcellset(grid, "SRF_$k"); psign[cid] = -1.0; end
@printf("  Plv_srf=%d  Pact_srf=%d  PlvPact_srf=%d  unloaded=%d\n",
        length(Plv_srf), length(Pact_srf), length(PlvPact_srf),
        length(getcellset(grid, "SRF_5")) + length(getcellset(grid, "SRF_13")))

# TODO (next): Phase-2 monolithic 0D Windkessel coupling (F_pact/F_plvpact + the
# implicit-Euler chamber block from limo_coupled_dynamic_strong.jl).

srf_id = zeros(Ferrite.getncells(grid))
for k in 1:n_srf
    for cid in getcellset(grid, "SRF_$k"); srf_id[cid] = k; end
end
# load_id: 1=endocardium(Plv), 2=actuator ext(Pact), 3=footprint(Plv−Pact), 0=unloaded
load_id = zeros(Ferrite.getncells(grid))
for c in endo_srf; load_id[c] = 1; end
for c in ext_srf;  load_id[c] = 2; end
for c in foot_srf; load_id[c] = 3; end
emark = zeros(Ferrite.getnnodes(grid));  for n in edge_nodes; emark[n] = 1.0; end

# HHT-α parameters (α = −0.3: strong high-frequency damping) + mass-proportional
# Rayleigh damping; adaptive Δt.  A Plv fill is ramped WITH the morph so the
# endocardium stays in tension (the pressure-free morph buckles the base rim).
N_dof = ndofs(dh); free = ch.free_dofs
α_hht = -0.3; γ_hht = 0.5 - α_hht; β_hht = (1 - α_hht)^2 / 4; α_damp = 10.0
Δt0 = 1e-3; Δt_min = 1e-6; Δt_max = 1e-2   # give up + save promptly once Δt collapses at the corner stall
tol = 1e-4; max_iter = 50
Pa2mmHg = 0.00750062
p_max   = 6.0 / Pa2mmHg    # Plv target at full morph [Pa]

K_int = allocate_matrix(dh); K_eff = allocate_matrix(dh); M = allocate_matrix(dh)
K_plv = allocate_matrix(dh); F_plv = zeros(N_dof)
r_int = zeros(N_dof); g_old = zeros(N_dof); res = zeros(N_dof)
δu = zeros(N_dof); u_trial = zeros(N_dof); rhs = zeros(N_dof)
a_new = zeros(N_dof); v_new = zeros(N_dof); Ma = zeros(N_dof); Mv = zeros(N_dof)
ũ = zeros(N_dof); ṽ = zeros(N_dof); u_new = zeros(N_dof)

n_e = ndofs_per_cell(dh); ke = zeros(n_e, n_e); re = zeros(n_e); u_e = zeros(n_e)
sdofs = Vector{Vector{Int}}(undef, Ferrite.getncells(grid))
for cell in CellIterator(dh); sdofs[Ferrite.cellid(cell)] = shelldofs(cell); end

# psign-weighted follower-pressure assembly over a cellset (Pc = psign[cid]), as in
# limo_dynamic_full.jl: assembles the unit-pressure load F (and tangent K) for a
# follower surface on the mirrored device.  `assemble_F!` also fills K; the `_res!`
# variant is load-only for the line search.
function assemble_F!(K_p, F_p, u, cellset)
    asm = start_assemble(K_p); fill!(F_p, 0.0)
    for cell in CellIterator(dh, cellset)
        cid = Ferrite.cellid(cell); sd = sdofs[cid]
        reinit!(scv, cell); @views u_e .= u[sd]
        fill!(ke, 0.0); fill!(re, 0.0)
        assemble_pressure!(re, scv, u_e, psign[cid])
        assemble_pressure_tangent!(ke, scv, u_e, psign[cid])
        assemble!(asm, sd, ke); @views F_p[sd] .+= re
    end
end
function assemble_F_res!(F_p, u, cellset)
    fill!(F_p, 0.0)
    for cell in CellIterator(dh, cellset)
        cid = Ferrite.cellid(cell); sd = sdofs[cid]
        reinit!(scv, cell); @views u_e .= u[sd]
        fill!(re, 0.0)
        assemble_pressure!(re, scv, u_e, psign[cid])
        @views F_p[sd] .+= re
    end
end

assemble_mass!(M, dh, scv, ρ, mat)
assemble_all!(K_int, r_int, dh, scv, zeros(N_dof), mat, sdofs, ke, re, u_e)
K_eff.nzval .= M.nzval .* (1/(β_hht*Δt0^2) + (1-α_hht)*α_damp*γ_hht/(β_hht*Δt0)) .+ (1-α_hht) .* K_int.nzval
apply_zero!(K_eff, r_int, ch)
F_lu = lu(K_eff)

# HHT-α Newton corrector (with Plv follower fill) + backtracking line search for
# one morph step; `u_new` updated in place; returns (converged, iters).
function solve_morph_step!(u_new, ũ, ṽ, p_new, Δt; max_iter=50, tol=1e-4)
    mfac = 1/(β_hht*Δt^2) + (1-α_hht)*α_damp*γ_hht/(β_hht*Δt)
    converged = false; iters = 0
    for iter in 1:max_iter
        iters = iter
        assemble_all!(K_int, r_int, dh, scv, u_new, mat, sdofs, ke, re, u_e)
        assemble_F!(K_plv, F_plv, u_new, Plv_srf)
        @. a_new = (u_new - ũ) / (β_hht*Δt^2)
        @. v_new = ṽ + (Δt*γ_hht) * a_new
        mul!(Ma, M, a_new); mul!(Mv, M, v_new)
        @. res = Ma + (1-α_hht)*(α_damp*Mv + r_int - p_new*F_plv) + α_hht*g_old
        apply_zero!(res, ch)
        res_norm = norm(@views res[free])
        res_norm < tol && (converged = true; break)
        K_eff.nzval .= M.nzval .* mfac .+ (1-α_hht) .* (K_int.nzval .- p_new .* K_plv.nzval)
        @. rhs = -res; apply_zero!(K_eff, rhs, ch)
        lu!(F_lu, K_eff); ldiv!(δu, F_lu, rhs)
        α_ls = 1.0; ls_ok = false
        for _ in 1:8
            @. u_trial = u_new + α_ls*δu; apply!(u_trial, ch)
            assemble_residual!(r_int, dh, scv, u_trial, mat, sdofs, re, u_e)
            assemble_F_res!(F_plv, u_trial, Plv_srf)
            @. a_new = (u_trial - ũ) / (β_hht*Δt^2)
            @. v_new = ṽ + (Δt*γ_hht) * a_new
            mul!(Ma, M, a_new); mul!(Mv, M, v_new)
            @. res = Ma + (1-α_hht)*(α_damp*Mv + r_int - p_new*F_plv) + α_hht*g_old
            apply_zero!(res, ch)
            (norm(@views res[free]) ≤ res_norm) && (ls_ok = true; break)
            α_ls /= 2
        end
        u_new .= u_trial; ls_ok || break
    end
    converged, iters
end

pvd = paraview_collection("minilimo-full")
vtk_step = Ref(0)
function write_frame(u, t)
    VTKGridFile("minilimo-full-$(vtk_step[])", dh) do vtk
        write_solution(vtk, dh, u)
        write_cell_data(vtk, srf_id, "SRF_id")
        write_cell_data(vtk, load_id, "load_id")
        write_cell_data(vtk, psign, "psign")
        Ferrite.write_node_data(vtk, emark, "edge")
        pvd[t] = vtk
    end
    vtk_step[] += 1
end

u = zeros(N_dof); apply!(u, ch); v = zeros(N_dof); a = zeros(N_dof)
write_frame(u, 0.0)

println("\nPHASE 1 — dynamic HHT-α morph (Plv fill ramps 0→$(round(p_max*Pa2mmHg,digits=1)) mmHg)")
@printf("%-6s  %-8s  %-8s  %-9s  %-6s  %-10s\n", "step", "t [s]", "λ", "Plv[mmHg]", "iters", "Δt")
un = zeros(N_dof)
let t = 0.0, step = 0, Δt_cur = Δt0, p = 0.0
@time while t < T_sim - 1e-10
    t_new = min(t + Δt_cur, T_sim)
    p_new = p_max * ramp(t_new)
    @. ũ = u + Δt_cur*v + (Δt_cur^2*(0.5-β_hht))*a
    @. ṽ = v + (Δt_cur*(1-γ_hht))*a
    u_new .= ũ; Ferrite.update!(ch, t_new); apply!(u_new, ch)
    converged, iters = solve_morph_step!(u_new, ũ, ṽ, p_new, Δt_cur; max_iter=max_iter, tol=tol)
    if converged
        step += 1
        @. a = (u_new - ũ) / (β_hht*Δt_cur^2)
        @. v = ṽ + (Δt_cur*γ_hht)*a
        mul!(Mv, M, v); @. g_old = α_damp*Mv + r_int - p_new*F_plv
        p = p_new; u .= u_new; t = t_new; Δt_cur = min(Δt_cur*1.2, Δt_max)
        if step % 4 == 0
            write_frame(u, t)
            @printf("%-6d  %-8.3f  %-8.4f  %-9.4f  %-6d  %-10.4e\n", step, t, ramp(t), p*Pa2mmHg, iters, Δt_cur)
        end
    else
        Δt_cur /= 2
        if Δt_cur < Δt_min
            @warn "minimum Δt reached at t=$(round(t,digits=4)) s (λ=$(round(ramp(t),digits=4))) — saving last converged state"
            write_frame(u, t)
            break
        end
    end
end
    un .= u
end
vtk_save(pvd)

using JLD2
jldsave(joinpath(@__DIR__, "minilimo_full_morph.jld2"); u=un)
println("saved minilimo_full_morph.jld2")
