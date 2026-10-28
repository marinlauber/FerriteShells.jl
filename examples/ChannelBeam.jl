# Thin-walled C-channel cantilever
#
# A cantilever with an open C (channel) cross-section, clamped at x = 0 and loaded at the
# tip x = L. Three load cases:
#
#   A. tip shear applied as the exact St-Venant shear flow  -> resultant passes through the
#      shear center, so the beam bends without twisting.  Reference: PL³/3EI (+ shear).
#   B. tip shear applied uniformly over the web             -> resultant acts on the web
#      centerline, an eccentricity e from the shear center, so the beam bends *and* twists.
#      Reference: Vlasov thin-walled torsion with restrained warping at the clamp.
#   C. case B continued into the large-deformation range with arc-length control, past the
#      limit point where the load the structure can carry starts to fall. No published
#      reference — no standard nonlinear shell benchmark uses an open thin-walled section —
#      so this is checked by mesh convergence and fold-treatment agreement instead.
#
# Case B is the demanding one of the two linear cases: the open section has a tiny St-Venant
# constant J, so almost all the torque is carried by warping, and the twist contributes more
# than a third of the tip deflection.
#
# Cross-section (midsurface), beam axis along x:
#
#        z
#        ^        y=0        y=b
#   h/2  |         +----------+   top flange
#        |         |
#        |         |  web
#        |         |
#  -h/2  |         +----------+   bottom flange
#        +-------------------------> y
#
# The two 90° folds at (y,z) = (0, ±h/2) are the interesting bit for a 5-DOF shell; the
# section "Mesh with split fold lines" below explains why and offers the exact alternative.

using FerriteShells
using LinearAlgebra
using Printf
using WriteVTK

# Geometry and material

const b_ch = 30.0     # flange width, measured from the web midline
const h_ch = 60.0     # web height, measured between flange midlines
const t_ch = 1.0      # wall thickness
const L_ch = 1200.0   # span (L/h = 20, L/t = 1200 -> thin and slender)
const E_ch = 2.1e5
const ν_ch = 0.3
const P_ch = 1.0      # tip transverse force, +z

const S_ch = 2b_ch + h_ch   # developed width of the section midline

# Thin-walled cross-section properties (midline theory)

"Second moment of area about the y-axis (strong axis, bending in the x-z plane)."
I_yy(b, h, t) = t * h^3 / 12 + b * t * h^2 / 2

"Shear-center offset from the web midline, on the side opposite the flanges (y_S = -e)."
shear_center(b, h) = 3b^2 / (h + 6b)

"St-Venant torsion constant of the open section."
J_sv(b, h, t) = (h + 2b) * t^3 / 3

"""
Warping constant about the shear center, ``I_ω = ∫ ω² t \\,ds``.

With ω = e·z on the web and ω = (h/2)(e - y) on the flanges this integrates to
`e² t h³/12 + t h² (e³ - (e-b)³)/6`, which equals the textbook
`b³h²t(2h+3b) / (12(h+6b))`.
"""
function warping_constant(b, h, t)
    e = shear_center(b, h)
    return e^2 * t * h^3 / 12 + t * h^2 * (e^3 - (e - b)^3) / 6
end

"Tip twist of a cantilever under a constant torque `T` with warping restrained at the clamp."
function vlasov_tip_twist(T, L, E, ν, b, h, t)
    G  = E / (2 * (1 + ν))
    GJ = G * J_sv(b, h, t)
    a  = sqrt(E * warping_constant(b, h, t) / GJ)   # warping decay length
    return T * (L - a * tanh(L / a)) / GJ
end

# Mesh
#
# The section is meshed in its developed (unrolled) coordinate `s`, so the grid is a plain
# structured rectangle in (s, x) that `shell_grid` folds into 3D. Nodes at the folds are
# shared by construction, and `s` is arc length so a uniform grid is uniform in space.
#
#   s ∈ [0, b]        top flange, from the free tip (y=b) inward to the web
#   s ∈ [b, b+h]      web, from z=+h/2 down to z=-h/2
#   s ∈ [b+h, 2b+h]   bottom flange, from the web outward to the free tip (y=b)

function channel_map(s, x, b, h)
    if s <= b
        return (x, b - s, h / 2)                # top flange
    elseif s <= b + h
        return (x, 0.0, h / 2 - (s - b))        # web
    else
        return (x, s - (b + h), -h / 2)         # bottom flange
    end
end

function channel_grid(n_s, n_x; b = b_ch, h = h_ch, L = L_ch)
    ## element boundaries must land on the folds, otherwise an element straddles a corner
    isinteger(n_s * b / (2b + h)) || throw(ArgumentError(
        "n_s = $n_s puts no element boundary at the folds s = $b, $(b+h); " *
        "need n_s * b / (2b + h) integral (for b=$b, h=$h: any multiple of 4)"))
    g2 = generate_grid(QuadraticQuadrilateral, (n_s, n_x),
                       Vec{2}((0.0, 0.0)), Vec{2}((2b + h, L)))
    grid = shell_grid(g2; map = n -> channel_map(n.x[1], n.x[2], b, h))
    addfacetset!(grid, "clamped", x -> x[1] ≈ 0.0)
    addfacetset!(grid, "tip",     x -> x[1] ≈ L)
    addnodeset!(grid, "tip_web",  x -> x[1] ≈ L && abs(x[2]) < 1e-8 && abs(x[3]) < 1e-8)
    addnodeset!(grid, "tip_top",  x -> x[1] ≈ L && abs(x[2] - b) < 1e-8 && x[3] > 0)
    addnodeset!(grid, "tip_bot",  x -> x[1] ≈ L && abs(x[2] - b) < 1e-8 && x[3] < 0)
    return grid
end

# Mesh with split fold lines
#
# `channel_grid` shares one node — and therefore one director — between the web and the
# flange all along a fold. A 5-DOF shell node carries only two rotations, but the web and
# the flange between them need three (about x̂, ŷ and ẑ; only the drilling rotation of each
# plate is absent), so the shared node is one rotation short and the fold is over-constrained.
#
# The exact alternative: mesh the three plates as separate strips so each fold line carries
# two coincident nodes, one per plate, each with its own plate normal. That gives 4 rotation
# DOFs where 3 are wanted, so one constraint per fold node pair is added back — the rotation
# about the fold tangent t̂ (here x̂) must agree between the two plates:
#
#     ω_A · t̂ = ω_B · t̂,   with   ω = -φ₂ T₁ + φ₁ T₂
#
# (the linearised Rodrigues director ḋ = φ₁T₁ + φ₂T₂ equals ω × G₃, and inverting that on
# the orthonormal frame (T₁,T₂,G₃) gives ω = -φ₂T₁ + φ₁T₂ up to the drilling component).
# The three translations are tied outright.

"Concatenate grids of one cell type, offsetting node indices; nodes are *not* deduplicated."
function merge_grids(grids)
    nodes = Node{3, Float64}[]
    cells = QuadraticQuadrilateral[]
    for g in grids
        off = length(nodes)
        append!(nodes, g.nodes)
        for c in getcells(g)
            push!(cells, QuadraticQuadrilateral(ntuple(i -> c.nodes[i] + off, 9)))
        end
    end
    return Grid(cells, nodes)
end

"""
    channel_grid_split(n_s, n_x) -> (grid, fold_pairs)

Same channel as [`channel_grid`](@ref), but meshed as three independent strips so the two
fold lines carry duplicated nodes. `fold_pairs` lists the coincident `(node_a, node_b)`
pairs, excluding those on the clamped end (already fixed by the Dirichlet BC there).
"""
function channel_grid_split(n_s, n_x; b = b_ch, h = h_ch, L = L_ch)
    isinteger(n_s * b / (2b + h)) || throw(ArgumentError(
        "n_s = $n_s does not split evenly into flange/web/flange strips"))
    n_f = round(Int, n_s * b / (2b + h))     # elements across one flange
    n_w = n_s - 2n_f                         # elements across the web
    strip(n, s₀, s₁) = shell_grid(
        generate_grid(QuadraticQuadrilateral, (n, n_x), Vec{2}((s₀, 0.0)), Vec{2}((s₁, L)));
        map = nd -> channel_map(nd.x[1], nd.x[2], b, h))
    grid = merge_grids([strip(n_f, 0.0, b), strip(n_w, b, b + h), strip(n_f, b + h, 2b + h)])

    ## the strips only touch on the fold lines, so any coincident node pair is a fold pair
    buckets = Dict{NTuple{3, Int}, Vector{Int}}()
    for (i, nd) in enumerate(getnodes(grid))
        key = ntuple(k -> round(Int, nd.x[k] * 1e6), 3)
        push!(get!(buckets, key, Int[]), i)
    end
    fold_pairs = Tuple{Int, Int}[]
    for ids in values(buckets)
        length(ids) == 1 && continue
        length(ids) == 2 || error("expected 2 coincident nodes on a fold, found $(length(ids))")
        getnodes(grid, ids[1]).x[1] ≈ 0.0 && continue   # clamped: Dirichlet already ties them
        push!(fold_pairs, (ids[1], ids[2]))
    end
    sort!(fold_pairs)

    addfacetset!(grid, "clamped", x -> x[1] ≈ 0.0)
    addfacetset!(grid, "tip",     x -> x[1] ≈ L)
    addnodeset!(grid, "tip_web",  x -> x[1] ≈ L && abs(x[2]) < 1e-8 && abs(x[3]) < 1e-8)
    addnodeset!(grid, "tip_top",  x -> x[1] ≈ L && abs(x[2] - b) < 1e-8 && x[3] > 0)
    addnodeset!(grid, "tip_bot",  x -> x[1] ≈ L && abs(x[2] - b) < 1e-8 && x[3] < 0)
    return grid, fold_pairs
end

"node id -> (u₁,u₂,u₃,φ₁,φ₂) dofs, in the interleaved `shelldofs` layout"
function node_dofmap(dh)
    map = Dict{Int, NTuple{5, Int}}()
    for cell in CellIterator(dh)
        sd = shelldofs(cell)
        for (I, nid) in enumerate(getnodes(cell))
            map[nid] = ntuple(k -> sd[5I - 5 + k], 5)
        end
    end
    return map
end

"""
    add_fold_constraints!(ch, dh, nf, fold_pairs, t̂)

Tie the coincident node pairs of a fold line whose tangent is `t̂`: three translation
constraints plus one equating the rotation about `t̂` on the two plates.
"""
function add_fold_constraints!(ch, dh, nf, fold_pairs, t̂::Vec{3})
    dofmap = node_dofmap(dh)
    for (a, bb) in fold_pairs
        da, db = dofmap[a], dofmap[bb]
        for k in 1:3
            add!(ch, Ferrite.AffineConstraint(db[k], [da[k] => 1.0], 0.0))
        end
        ## ω·t̂ = -φ₂ (T₁·t̂) + φ₁ (T₂·t̂), one row over the four φ dofs
        c = (a => (-(nf.T₁[a] ⋅ t̂), nf.T₂[a] ⋅ t̂), bb => ((nf.T₁[bb] ⋅ t̂), -(nf.T₂[bb] ⋅ t̂)))
        coeffs = [da[5] => c[1][2][1], da[4] => c[1][2][2],
                  db[5] => c[2][2][1], db[4] => c[2][2][2]]
        i = argmax(abs(p.second) for p in coeffs)
        piv = coeffs[i]
        abs(piv.second) > 1e-8 || error("degenerate fold constraint at nodes ($a, $bb)")
        masters = [p.first => -p.second / piv.second for (j, p) in enumerate(coeffs) if j != i]
        add!(ch, Ferrite.AffineConstraint(piv.first, masters, 0.0))
    end
    return ch
end

# Tip tractions
#
# Both load cases are line loads (force per unit length) on the tip cross-section edge,
# given as a function of the 3D position.

"""
Exact St-Venant shear flow for a transverse shear `V` in +z, as a traction vector.

`q(s) = -(V/I) Q(s)` along the midline, with `Q(s)` the first moment about the y-axis of
the material between the free edge and `s`. Its resultant is `V ẑ` acting through the
shear center, so it loads the beam in pure bending.
"""
function shear_flow_traction(V, b, h, t)
    I = I_yy(b, h, t)
    return function (x::Vec{3})
        y, z = x[2], x[3]
        if abs(abs(z) - h / 2) < 1e-8 && y > 1e-8            # flange
            q_y = sign(z) * (V / I) * t * (h / 2) * (b - y)
            return Vec{3}((0.0, q_y, 0.0))
        else                                                 # web
            Q = t * (h * b / 2 + (h^2 / 4 - z^2) / 2)
            return Vec{3}((0.0, 0.0, (V / I) * Q))
        end
    end
end

"Uniform transverse load `V` in +z spread over the web only; resultant on the web centerline."
function web_traction(V, b, h)
    return function (x::Vec{3})
        abs(x[2]) < 1e-8 || return zero(Vec{3})
        return Vec{3}((0.0, 0.0, V / h))
    end
end

# Solver

"Grid, DOFs, constraints and cell values for one mesh and fold treatment."
struct ChannelModel
    grid; dh; ch; scv; mat; ip
end

"Build the model for an `n_s` x `n_x` mesh with the given `fold` treatment."
function channel_model(n_s, n_x; fold::Symbol = :averaged)
    fold in (:averaged, :split) || throw(ArgumentError("fold must be :averaged or :split"))
    grid, fold_pairs = fold === :split ? channel_grid_split(n_s, n_x) :
                                         (channel_grid(n_s, n_x), Tuple{Int,Int}[])
    ip   = Lagrange{RefQuadrilateral, 2}()
    mat  = LinearElastic(E_ch, ν_ch, t_ch)
    nf   = NodeFrames(grid, ip)
    scv  = ShellCellValues(QuadratureRule{RefQuadrilateral}(3), ip, ip; mitc = MITC9, frames = nf)

    dh = DofHandler(grid); add!(dh, :u, ip^3); add!(dh, :θ, ip^2); close!(dh)

    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "clamped"), x -> zeros(3), [1, 2, 3]))
    add!(ch, Dirichlet(:θ, getfacetset(grid, "clamped"), x -> zeros(2), [1, 2]))
    add_fold_constraints!(ch, dh, nf, fold_pairs, Vec{3}((1.0, 0.0, 0.0)))
    close!(ch); Ferrite.update!(ch, 0.0)
    return ChannelModel(grid, dh, ch, scv, mat, ip)
end

"Consistent nodal load vector for a tip line load."
function tip_load(m::ChannelModel, traction)
    f = zeros(ndofs(m.dh))
    assemble_traction!(f, m.dh, getfacetset(m.grid, "tip"), m.ip,
                       FacetQuadratureRule{RefQuadrilateral}(3), traction)
    return f
end

"Tangent `K` and internal force `r` of the (geometrically nonlinear) RM shell at `u`."
function assemble_tangent!(K, r, m::ChannelModel, u)
    n_e = ndofs_per_cell(m.dh); ke = zeros(n_e, n_e); re = zeros(n_e)
    asm = start_assemble(K, r)
    for cell in CellIterator(m.dh)
        fill!(ke, 0.0); fill!(re, 0.0)
        reinit!(m.scv, cell)
        ue = u[shelldofs(cell)]
        membrane_tangent_RM!(ke, m.scv, ue, m.mat); membrane_residuals_RM!(re, m.scv, ue, m.mat)
        bending_tangent_RM!(ke, m.scv, ue, m.mat);  bending_residuals_RM!(re, m.scv, ue, m.mat)
        assemble!(asm, shelldofs(cell), ke, re)
    end
end

"""
    channel_solve(n_s, n_x; traction, fold = :averaged) -> (u, dh, grid)

Assemble and solve the linear elastic channel with the given tip traction.

`fold` selects the treatment of the two 90° corners: `:averaged` shares one node (and one
area-averaged, 45°-bisector director) between web and flange; `:split` gives each plate its
own node and director and ties them with [`add_fold_constraints!`](@ref).
"""
function channel_solve(n_s, n_x; traction, fold::Symbol = :averaged)
    m = channel_model(n_s, n_x; fold)
    K = allocate_matrix(m.dh, m.ch)
    r = zeros(ndofs(m.dh))
    assemble_tangent!(K, r, m, zeros(ndofs(m.dh)))   # at u = 0 this is the linear stiffness
    f = tip_load(m, traction)
    apply!(K, f, m.ch)
    u = K \ f
    apply!(u, m.ch)
    return u, m.dh, m.grid
end

"Displacement `Vec{3}` of the single node in `nodeset`."
function node_displacement(u, dh, grid, nodeset)
    nid = first(getnodeset(grid, nodeset))
    for cell in CellIterator(dh)
        sd = shelldofs(cell)
        for (I, g) in enumerate(getnodes(cell))
            g == nid && return Vec{3}((u[sd[5I-4]], u[sd[5I-3]], u[sd[5I-2]]))
        end
    end
    error("node set $nodeset is empty or not found")
end

# Case C: arc-length continuation past the limit point
#
# Under the web load the channel bends *and* twists, and the twist feeds back on itself: the
# more the section rotates, the larger the lever arm of the load about the shear center. Past
# roughly 55° of tip twist that feedback wins and the structure reaches a limit point — the
# load it can carry starts to fall. Load control cannot go past it (Newton simply stops
# converging), so the path is traced with a cylindrical arc-length constraint instead, which
# lets the load parameter decrease.
#
# There is no published reference for this path — no standard nonlinear benchmark uses an open
# thin-walled section — so what is checked is mesh convergence of the limit load and agreement
# between the two fold treatments.

"""
    condense(K, ch, rhss...) -> (K_condensed, condensed_rhss)

Condense the same tangent against several right-hand sides. `apply_zero!(K, f, ch)` edits `K`
and `f` together and its effect on `K` does not depend on `f`, so condensing a copy per rhs and
keeping one of the (identical) matrices is correct — and lets the arc-length corrector, which
needs both `K⁻¹g` and `K⁻¹F`, get away with a single factorization.
"""
function condense(K, ch, rhss...)
    Kc = nothing
    out = map(rhss) do f
        Kt = copy(K); fc = copy(f)
        apply_zero!(Kt, fc, ch)
        Kc === nothing && (Kc = Kt)
        fc
    end
    return Kc, out
end

"""
    arclength(m, F_ext; Δl0, n_steps, probe, ...) -> Vector{NamedTuple}

Cylindrical arc-length (Crisfield) continuation of `r(u) = λ F_ext`.

Each corrector iteration solves against both the out-of-balance force and `F_ext`, then picks
`δλ` from the roots of ``‖Δu + δū + δλ\\,δu_t‖² = Δl²``, taking the root that keeps travelling
in the direction of the current increment. The predictor sign follows `sign(Δu_prev · δu_t)`,
which flips exactly when the tangent changes definiteness — that is what carries the path
through the limit point. The arc length adapts toward four corrector iterations per step and
halves on a failed step.

Each entry of the returned path is `(; λ, probe(u, m)...)`.
"""
function arclength(m::ChannelModel, F_ext; Δl0, n_steps, probe, tol = 1e-6, max_iter = 20,
                   Δl_min = 1e-4, stop_after_limit = 0.0, store_u = false)
    N = ndofs(m.dh)
    K = allocate_matrix(m.dh, m.ch); r = zeros(N)
    u = zeros(N); λ = 0.0; Δl = Δl0
    Δu_prev = zeros(N)
    path = NamedTuple[]
    λ_max = -Inf
    step = 0
    while step < n_steps
        step += 1
        ## predictor: tangent direction, scaled to the arc length
        assemble_tangent!(K, r, m, u)
        Kc, (F_c,) = condense(K, m.ch, F_ext)
        δu_t = lu(Kc) \ F_c; apply_zero!(δu_t, m.ch)
        s  = step == 1 ? 1.0 : sign(dot(Δu_prev, δu_t))
        s == 0.0 && (s = 1.0)
        Δλ = s * Δl / norm(δu_t)
        Δu = Δλ .* δu_t
        ## corrector
        converged = false; n_it = 0
        for it in 1:max_iter
            assemble_tangent!(K, r, m, u .+ Δu)
            g = (λ + Δλ) .* F_ext .- r
            Kc, (g_c, F_c) = condense(K, m.ch, g, F_ext)
            if norm(g_c) < tol * max(1.0, norm(F_ext))
                converged = true; n_it = it - 1; break
            end
            n_it = it
            fact = lu(Kc)
            δū   = fact \ g_c; apply_zero!(δū,   m.ch)
            δu_t = fact \ F_c; apply_zero!(δu_t, m.ch)
            w = Δu .+ δū
            a = dot(δu_t, δu_t); b = 2 * dot(w, δu_t); c = dot(w, w) - Δl^2
            disc = b^2 - 4a * c
            disc < 0 && (n_it = max_iter + 1; break)      # no intersection: shrink the step
            rt = sqrt(disc)
            δλ1, δλ2 = (-b + rt) / (2a), (-b - rt) / (2a)
            δλ = dot(w .+ δλ1 .* δu_t, Δu) > dot(w .+ δλ2 .* δu_t, Δu) ? δλ1 : δλ2
            Δu = w .+ δλ .* δu_t
            Δλ += δλ
        end
        if !converged
            Δl /= 2
            Δl < Δl_min && break
            step -= 1
            continue
        end
        u .+= Δu; λ += Δλ; Δu_prev = copy(Δu)
        push!(path, (; λ, probe(u, m)..., u = store_u ? copy(u) : nothing))
        λ_max = max(λ_max, λ)
        Δl *= clamp(sqrt(4 / max(n_it, 1)), 0.7, 1.4)     # target ~4 corrector iterations
        stop_after_limit > 0 && λ < (1 - stop_after_limit) * λ_max && break
    end
    return path
end

"Tip web deflection, lateral drift and twist, as a probe for the continuation path."
function tip_probe(u, m::ChannelModel)
    uw = node_displacement(u, m.dh, m.grid, "tip_web")
    ut = node_displacement(u, m.dh, m.grid, "tip_top")
    ub = node_displacement(u, m.dh, m.grid, "tip_bot")
    return (; u_z = uw[3], u_y = uw[2], θ = (ub[2] - ut[2]) / h_ch)
end

"Index of the first local maximum of the load parameter along the path, or `nothing`."
function first_limit_point(path)
    for i in 2:length(path)-1
        path[i].λ > path[i-1].λ && path[i].λ ≥ path[i+1].λ && return i
    end
    return nothing
end

"""
    channel_limit_load(n_s, n_x; fold, P_ref) -> (P_limit, θ_limit, path)

Trace the web-load path with arc-length continuation and return the first limit point.
"""
function channel_limit_load(n_s, n_x; fold = :split, P_ref = 1000.0,
                            Δl0 = 5.0, n_steps = 120, kwargs...)
    m    = channel_model(n_s, n_x; fold)
    F    = tip_load(m, web_traction(P_ref, b_ch, h_ch))
    path = arclength(m, F; Δl0, n_steps, probe = tip_probe, kwargs...)
    i    = first_limit_point(path)
    i === nothing && return (NaN, NaN, path)
    return (path[i].λ * P_ref, path[i].θ, path)
end

"""
    write_case_C_vtk(n_s = 8, n_x = 16; fold = :split, every = 2)

Write the arc-length path of case C as a ParaView collection `channel_limit_path.pvd`, with
the load parameter as the time value, so the deformed shape can be scrubbed through the limit
point. Unlike cases A and B these displacements are at full scale — do **not** warp them.

Also reports how far the response has left the linear regime at the limit point.
"""
function write_case_C_vtk(n_s = 8, n_x = 16; fold = :split, every = 2, P_ref = 1000.0)
    m    = channel_model(n_s, n_x; fold)
    F    = tip_load(m, web_traction(P_ref, b_ch, h_ch))
    path = arclength(m, F; Δl0 = 5.0, n_steps = 120, probe = tip_probe,
                     stop_after_limit = 0.02, store_u = true)
    i    = first_limit_point(path)
    pvd  = paraview_collection("channel_limit_path")
    VTKGridFile("channel_limit_path-0", m.dh) do vtk
        write_solution(vtk, m.dh, zeros(ndofs(m.dh))); pvd[0.0] = vtk
    end
    n_frames = 1
    for (k, p) in enumerate(path)
        (k % every == 0 || k == i) || continue
        VTKGridFile("channel_limit_path-$k", m.dh) do vtk
            write_solution(vtk, m.dh, p.u); pvd[p.λ * P_ref] = vtk
        end
        n_frames += 1
    end
    close(pvd)
    @printf("wrote channel_limit_path.pvd (%d frames, %s folds, %d x %d)\n",
            n_frames, fold, n_s, n_x)
    if i !== nothing
        p, P = path[i], path[i].λ * P_ref
        @printf("  limit point P = %.1f: u_z = %.1f (%.1f%% of L), u_y = %.1f (%.1f%%), θ = %.1f deg\n",
                P, p.u_z, 100p.u_z / L_ch, p.u_y, 100p.u_y / L_ch, rad2deg(p.θ))
        @printf("  linear theory at the same load: u_z = %.1f, θ = %.1f deg",
                δ_web_ref * P, rad2deg(θ_ref * P))
        @printf("  -> nonlinear/linear = %.2fx (u_z), %.2fx (θ)\n",
                p.u_z / (δ_web_ref * P), p.θ / (θ_ref * P))
        e = path[end]
        @printf("  end of traced path P = %.1f: u_z = %.1f (%.1f%% of L), θ = %.1f deg\n",
                e.λ * P_ref, e.u_z, 100e.u_z / L_ch, rad2deg(e.θ))
    end
    return path
end

# Reference values

const e_sc   = shear_center(b_ch, h_ch)
const I_ref  = I_yy(b_ch, h_ch, t_ch)
const G_ch   = E_ch / (2 * (1 + ν_ch))

# Case A: bending only. Timoshenko correction assumes the web carries the shear.
const δ_euler     = P_ch * L_ch^3 / (3 * E_ch * I_ref)
const δ_shear     = P_ch * L_ch / (G_ch * h_ch * t_ch)
const δ_bend_ref  = δ_euler + δ_shear

# Case B: bending + warping torsion about the shear center.
const θ_ref       = vlasov_tip_twist(P_ch * e_sc, L_ch, E_ch, ν_ch, b_ch, h_ch, t_ch)
const δ_web_ref   = δ_bend_ref + e_sc * θ_ref

function report_section()
    @printf("Section: b=%.1f h=%.1f t=%.1f L=%.1f  E=%.4g ν=%.2f\n",
            b_ch, h_ch, t_ch, L_ch, E_ch, ν_ch)
    @printf("  I_yy = %.6g   J = %.6g   I_ω = %.6g   e_shear = %.6g\n",
            I_ref, J_sv(b_ch, h_ch, t_ch), warping_constant(b_ch, h_ch, t_ch), e_sc)
    @printf("  reference A: w_tip = %.6e  (Euler %.6e + shear %.6e)\n",
            δ_bend_ref, δ_euler, δ_shear)
    @printf("  reference B: θ_tip = %.6e  w_web = %.6e\n", θ_ref, δ_web_ref)
end

"Tip deflection under the shear-center load (case A)."
function channel_bending(n_s, n_x; fold = :averaged)
    u, dh, grid = channel_solve(n_s, n_x; fold,
                                traction = shear_flow_traction(P_ch, b_ch, h_ch, t_ch))
    return node_displacement(u, dh, grid, "tip_web")[3]
end

"Tip web deflection and tip twist under the web load (case B)."
function channel_bending_torsion(n_s, n_x; fold = :averaged)
    u, dh, grid = channel_solve(n_s, n_x; fold, traction = web_traction(P_ch, b_ch, h_ch))
    w   = node_displacement(u, dh, grid, "tip_web")[3]
    u_t = node_displacement(u, dh, grid, "tip_top")
    u_b = node_displacement(u, dh, grid, "tip_bot")
    θ   = (u_b[2] - u_t[2]) / h_ch     # antisymmetric part -> bending contribution cancels
    return w, θ
end

function run_case_A(meshes = [(4, 8), (8, 16), (16, 32)])
    println("\n--- Case A: tip shear through the shear center (pure bending) ---")
    println("reference w_tip = $(@sprintf("%.6e", δ_bend_ref))  (Euler-Bernoulli + Timoshenko shear)")
    @printf("%6s %6s | %14s %8s | %14s %8s\n",
            "n_s", "n_x", "w (averaged)", "err", "w (split)", "err")
    for (n_s, n_x) in meshes
        wa = channel_bending(n_s, n_x; fold = :averaged)
        ws = channel_bending(n_s, n_x; fold = :split)
        @printf("%6d %6d | %14.6e %7.2f%% | %14.6e %7.2f%%\n", n_s, n_x,
                wa, 100 * (wa - δ_bend_ref) / δ_bend_ref,
                ws, 100 * (ws - δ_bend_ref) / δ_bend_ref)
    end
end

function run_case_B(meshes = [(4, 8), (8, 16), (16, 32)])
    println("\n--- Case B: tip shear on the web (bending + warping torsion) ---")
    println("reference w_web = $(@sprintf("%.6e", δ_web_ref)), θ_tip = $(@sprintf("%.6e", θ_ref))  (Vlasov)")
    @printf("%6s %6s | %14s %8s %8s | %14s %8s %8s\n",
            "n_s", "n_x", "w (averaged)", "err", "θ err", "w (split)", "err", "θ err")
    for (n_s, n_x) in meshes
        wa, θa = channel_bending_torsion(n_s, n_x; fold = :averaged)
        ws, θs = channel_bending_torsion(n_s, n_x; fold = :split)
        @printf("%6d %6d | %14.6e %7.2f%% %7.2f%% | %14.6e %7.2f%% %7.2f%%\n", n_s, n_x,
                wa, 100 * (wa - δ_web_ref) / δ_web_ref, 100 * (θa - θ_ref) / θ_ref,
                ws, 100 * (ws - δ_web_ref) / δ_web_ref, 100 * (θs - θ_ref) / θ_ref)
    end
end

"""
    run_case_C(meshes = [(8, 16)])

Locate the limit point of the web-load path on each mesh, for both fold treatments.

The default is the single coarsest mesh that resolves a genuine peak (8x12 flattens it into a
0.6% plateau); pass `[(8, 16), (16, 32)]` for the convergence study, which takes a few minutes.
Reference values from that study: 8x16 gives 322.3 (`:averaged`) / 333.9 (`:split`), 16x32
gives 324.6 / 329.5 — the two treatments close from 11.6 apart to 4.9 under refinement.
"""
function run_case_C(meshes = [(8, 16)])
    println("\n--- Case C: web load continued past the limit point (arc-length) ---")
    println("no published reference; checking mesh convergence and fold-treatment agreement")
    @printf("%6s %6s | %12s %9s | %12s %9s\n",
            "n_s", "n_x", "P_lim (avg)", "θ (deg)", "P_lim (split)", "θ (deg)")
    for (n_s, n_x) in meshes
        Pa, θa, _ = channel_limit_load(n_s, n_x; fold = :averaged, stop_after_limit = 0.02)
        Ps, θs, _ = channel_limit_load(n_s, n_x; fold = :split,    stop_after_limit = 0.02)
        @printf("%6d %6d | %12.2f %9.2f | %12.2f %9.2f\n",
                n_s, n_x, Pa, rad2deg(θa), Ps, rad2deg(θs))
    end
end

"Print the load-twist path of case C, so the descending branch is visible."
function print_path(n_s = 8, n_x = 16; fold = :split, every = 4)
    _, _, path = channel_limit_load(n_s, n_x; fold)
    P_ref = 1000.0
    @printf("\n%5s %10s %10s %10s %9s\n", "step", "P", "u_z", "u_y", "θ (deg)")
    for (i, p) in enumerate(path)
        i % every == 0 && @printf("%5d %10.2f %10.3f %10.3f %9.2f\n",
                                  i, p.λ * P_ref, p.u_z, p.u_y, rad2deg(p.θ))
    end
end

"""
    write_vtk(n_s = 16, n_x = 32; fold = :split)

Solve both load cases on one mesh and write `channel_bending.vtu` (case A) and
`channel_torsion.vtu` (case B). Each file carries the `u` and `θ` fields.

The tip deflection is ~4e-2 on a beam 1200 long, so open the files with a *Warp By Vector*
filter on `u` and a scale factor around `10^4` to see the deformed shape at all — case B
then shows the twist that case A does not have.
"""
function write_vtk(n_s = 16, n_x = 32; fold = :split)
    cases = ("channel_bending" => shear_flow_traction(P_ch, b_ch, h_ch, t_ch),
             "channel_torsion" => web_traction(P_ch, b_ch, h_ch))
    for (name, tr) in cases
        u, dh, _ = channel_solve(n_s, n_x; traction = tr, fold)
        VTKGridFile(name, dh) do vtk
            write_solution(vtk, dh, u)
        end
        @printf("wrote %s.vtu  (%s folds, %d x %d)\n", name, fold, n_s, n_x)
    end
end

# run
report_section()
run_case_A()
run_case_B()
run_case_C()
println()
write_vtk()
