# Shared miniLIMO geometry, initial-condition, and assembly helpers.
#
# Every script in this directory builds the same rectangular multi-surface Q9
# mesh (`make_minilimo_grid`), morphs its flat `y=0` edge onto the target
# elliptic arc (`map_initial`/`find_points`/`bisect` + `generate_boundary_function`),
# and assembles the same RM membrane+bending residual/tangent, mass, and
# follower-pressure operators.  Those shared pieces live here; each script only
# adds its own solver / loading strategy.  Include with
# `include(joinpath(@__DIR__, "util.jl"))`.
#
# `make_minilimo_grid` takes `even_Np=true` for the antisymmetric full-device
# mesh; `generate_boundary_function` takes a `ramp` kwarg selecting the edge
# load schedule (defaults to a frozen full morph).

using FerriteShells, LinearAlgebra, QuadGK

function color(vtk, grid, cellset)
    z = zeros(Ferrite.getncells(grid))
    z[collect(Ferrite.getcellset(grid, cellset))] .= 1.0
    write_cell_data(vtk, z, cellset)
end

function bisect(f, θ_lo, θ_hi; tolerance=1e-8)
    θ_mid = (θ_lo + θ_hi) / 2
    while θ_hi - θ_lo > tolerance
        θ_mid = (θ_lo + θ_hi) / 2
        f(θ_mid) * f(θ_lo) < 0 ? (θ_hi = θ_mid) : (θ_lo = θ_mid)
    end
    return θ_mid
end

function find_points(x, y, A, B, L)
    N = length(x)
    x_new = similar(x); y_new = similar(y)
    x_min = minimum(x)
    for i in (1, N)
        θ = (x[i] - x_min) * π / L
        x_new[i] = -A * cos(θ); y_new[i] = -B * sin(θ)
    end
    lengths = @views sqrt.((x[2:end] .- x[1:end-1]).^2 .+ (y[2:end] .- y[1:end-1]).^2)
    θ0 = 0.0
    for i in 1:N-2
        x0, y0, d = x_new[N-i+1], y_new[N-i+1], lengths[N-i]
        θ0 = bisect(θ0, π) do θ
            sqrt((A*cos(θ)-x0)^2 + (B*sin(θ)-y0)^2) - d
        end
        x_new[N-i] = A*cos(θ0); y_new[N-i] = B*sin(θ0)
    end
    x_new, y_new
end

function map_initial(x, y, Ar)
    L = maximum(x) - minimum(x)
    ds(θ, a) = sqrt(a^2*sin(θ)^2 + (a/Ar)^2*cos(θ)^2)
    find_a(a) = quadgk(θ -> ds(θ, a), 0, π)[1] - L
    a0 = bisect(find_a, 0.0, L)
    a = bisect(0.98*a0, 1.08*a0) do a
        xi, yi = find_points(x, y, a, a/Ar, L)
        @views sum(sqrt.((xi[2:end].-xi[1:end-1]).^2 .+ (yi[2:end].-yi[1:end-1]).^2)) - L
    end
    find_points(x, y, a, a/Ar, L)
end

# Rectangular approximation of the miniLIMO geometry without rounded edges.
# SRF_1: outer endocardium (Plv only)
# SRF_2: inner endocardium at actuator footprint (Plv − Pact)
# SRF_3: actuator exterior shell (Pact only), double-layer with SRF_2.
#
# `order` selects the element: `2` → Q9 (`QuadraticQuadrilateral`, MITC9, default),
# `1` → Q4 (`Quadrilateral`, MITC4).  The node grid is refined `order` times per
# element edge; all the `(px, py)` bookkeeping runs in this refined index space so
# a cell spans `order` index units.  Cellsets/nodesets/facetsets are identical
# across orders — only the element type and node density change.
function make_minilimo_grid(;
    nx_left=3, nx_act=10, nx_right=3,
    ny_bot=1, ny_act=14, ny_top=2,
    W=0.10118, H=0.109, x_act=0.035, y_lo=0.004, y_hi=0.09,
    Np=1, even_Np=false, order=2, grade_bot=1.0)

    @assert order in (1, 2) "order must be 1 (Q4/MITC4) or 2 (Q9/MITC9)"
    @assert nx_act % Np == 0 "nx_act ($nx_act) must be divisible by Np ($Np)"
    even_Np && @assert iseven(Np) "full-device antisymmetric loading needs even Np so pouches do not straddle x=0"
    nx_per_pouch = nx_act ÷ Np

    # Geometric grading of a [a,b] strip into n intervals: ratio r>1 clusters nodes
    # toward y=a (the morphed y=0 edge) so the mesh can resolve the wrinkle that
    # relieves the fold-induced corner compression; r=1 is uniform.
    function graded(a, b, n, r)
        r ≈ 1.0 && return collect(range(a, b, n + 1))
        w = [r^k for k in 0:n-1];  w ./= sum(w)
        a .+ (b - a) .* vcat(0.0, cumsum(w))
    end

    Lx = W / 2
    xs = vcat(range(-Lx,   -x_act, nx_left + 1),
              range(-x_act,  x_act, nx_act  + 1)[2:end],
              range( x_act,    Lx,  nx_right + 1)[2:end])
    ys = vcat(graded(0.0,  y_lo, ny_bot, grade_bot),
              range(y_lo, y_hi, ny_act + 1)[2:end],
              range(y_hi,   H,  ny_top + 1)[2:end])
    nx = length(xs) - 1;  ny = length(ys) - 1

    # Refine each segment into `order` sub-intervals (order=2 inserts edge midpoints
    # for Q9; order=1 is the identity for Q4).
    function refine(v, order)
        order == 1 && return collect(float.(v))
        w = Vector{float(eltype(v))}(undef, order*(length(v)-1) + 1)
        for i in 1:length(v)-1, k in 0:order-1
            w[order*(i-1)+k+1] = v[i] + (v[i+1]-v[i])*k/order
        end
        w[end] = v[end];  w
    end
    xs_f = refine(xs, order);  ys_f = refine(ys, order)

    endo_node(px, py) = py * (order*nx + 1) + px + 1
    n_endo = (order*nx + 1) * (order*ny + 1)
    endo_coords = [Vec{3}((xs_f[px+1], ys_f[py+1], 0.0)) for py in 0:order*ny for px in 0:order*nx]

    py_lo = order*ny_bot;  py_hi = order*(ny_bot + ny_act)

    act_nodes = [Dict{Tuple{Int,Int},Int}() for _ in 1:Np]
    act_coords = Vec{3,Float64}[]
    for p in 1:Np
        px_lo_p = order*(nx_left + (p-1)*nx_per_pouch)
        px_hi_p = order*(nx_left + p*nx_per_pouch)
        for py in py_lo:py_hi, px in px_lo_p:px_hi_p
            if px == px_lo_p || px == px_hi_p || py == py_lo || py == py_hi
                act_nodes[p][(px, py)] = endo_node(px, py)
            else
                push!(act_coords, Vec{3}((xs_f[px+1], ys_f[py+1], 0.0)))
                act_nodes[p][(px, py)] = n_endo + length(act_coords)
            end
        end
    end

    # Build one element from a node-id accessor `nid(dx, dy)` over the refined
    # index space; corner-then-edge-then-centre ordering for Q9, corners for Q4.
    function make_cell(nid)
        order == 2 ?
            QuadraticQuadrilateral((
                nid(0,0), nid(2,0), nid(2,2), nid(0,2),
                nid(1,0), nid(2,1), nid(1,2), nid(0,1), nid(1,1))) :
            Quadrilateral((nid(0,0), nid(1,0), nid(1,1), nid(0,1)))
    end
    elem_endo(px, py)    = make_cell((dx, dy) -> endo_node(px+dx, py+dy))
    elem_act(p, px, py)  = make_cell((dx, dy) -> act_nodes[p][(px+dx, py+dy)])
    CellT = order == 2 ? QuadraticQuadrilateral : Quadrilateral

    srf1   = Int[]
    srf2_k = [Int[] for _ in 1:Np]
    endo_cells = CellT[]
    for iy in 0:ny-1, ix in 0:nx-1
        push!(endo_cells, elem_endo(order*ix, order*iy))
        cid    = length(endo_cells)
        ix_rel = ix - nx_left
        if iy >= ny_bot && iy < ny_bot + ny_act && ix_rel >= 0 && ix_rel < nx_act
            push!(srf2_k[ix_rel ÷ nx_per_pouch + 1], cid)
        else
            push!(srf1, cid)
        end
    end
    n_ec = length(endo_cells)

    srf3_k = [Int[] for _ in 1:Np]
    act_cells = CellT[]
    for p in 1:Np
        for k in 1:nx_per_pouch
            ix = nx_left + (p-1)*nx_per_pouch + k - 1
            px = order*ix
            for iy in ny_bot:ny_bot+ny_act-1
                push!(act_cells, elem_act(p, px, order*iy))
                push!(srf3_k[p], n_ec + length(act_cells))
            end
        end
    end

    grid = Grid(vcat(endo_cells, act_cells), Node.(vcat(endo_coords, act_coords)))
    addcellset!(grid, "SRF_1", Set(srf1))
    srf2_all = Int[];  srf3_all = Int[]
    for k in 1:Np
        addcellset!(grid, "SRF_2_$k", Set(srf2_k[k]))
        addcellset!(grid, "SRF_3_$k", Set(srf3_k[k]))
        append!(srf2_all, srf2_k[k]);  append!(srf3_all, srf3_k[k])
    end
    addcellset!(grid, "SRF_2", Set(srf2_all))
    addcellset!(grid, "SRF_3", Set(srf3_all))
    addnodeset!(grid, "edge", x -> x[2] ≈ 0.0)
    addfacetset!(grid, "sym", x -> x[2] ≈ H || abs(x[1]) ≈ Lx)
    # Orientation-split symmetry facets for single-component director clamps:
    # the director must stay in the symmetry plane, so only the out-of-plane
    # rotation is fixed (φ₁ on x=±Lx, φ₂ on y=H) — the in-plane fold rotation is free.
    addfacetset!(grid, "sym_x", x -> abs(x[1]) ≈ Lx)   # x=±Lx planes → fix φ₁
    addfacetset!(grid, "sym_y", x -> x[2] ≈ H)         # y=H plane   → fix φ₂
    return grid
end

# `corner_relief` tapers the morph amplitude smoothly to zero over the first/last
# `corner_relief` edge nodes (cosine blend: 0 at the x=±Lx corner → 1 at node
# corner_relief+1 inward).  This gives the singular edge∩sym corner a small flat
# relief facet so the fold-induced contraction is carried as benign membrane
# tension instead of the compression singularity that snaps in Phase 2.  A hard
# step (fully-flat nodes) makes a worse kink one element in; the cosine taper
# avoids that.  Response is non-monotonic in width — validate per mesh/material;
# corner_relief=3 flips the corner to mild tension on the Np=3 order-1 mesh.
# corner_relief=0 is the exact ellipse morph.
# `zsign` flips the out-of-plane fold direction (u_z = zsign·ellipse height): a
# closed pouch's two coincident sheets fold to opposite z (SRF_1 up, SRF_9 down)
# so it opens during the morph.  In-plane Δx is unaffected (both rims contract the
# same way, staying coincident in x,y while separating in z).
function generate_boundary_function(grid, nodeset; ramp = t -> 1.0, corner_relief = 0, zsign = 1)
    top_nodes = get_node_coordinate.(getnodes(grid, nodeset))
    idx = sortperm(top_nodes)
    node_sorted = top_nodes[idx]
    # Dedupe coincident nodes (e.g. the two stacked sheets sharing a base rim):
    # map_initial needs distinct arc positions, and the nearest-node closure below
    # re-covers the duplicates.  No-op when the edge nodes are already distinct.
    keep = [i for i in eachindex(node_sorted) if i == 1 || !(node_sorted[i] ≈ node_sorted[i-1])]
    node_sorted = node_sorted[keep]
    Ar = 80.2 / 55.2
    x, y = getindex.(node_sorted, 1), getindex.(node_sorted, 2)
    x_new, y_new = map_initial(x, y, Ar)
    Xs = vcat(x', y'); dXs = vcat(x_new' .- x', zsign .* y_new')
    if corner_relief > 0
        n = min(corner_relief, (length(x) - 1) ÷ 2)
        for j in 0:n
            wj = 0.5 * (1 - cos(π * j / n))
            @views dXs[:, 1 + j]   .*= wj
            @views dXs[:, end - j] .*= wj
        end
    end
    return function prescribed_u(x, t)
        idx = findmin(dropdims(sum(abs2, Xs .- [x[1], x[2]], dims=1), dims=1))[2]
        return ramp(t) .* dXs[:, idx]
    end
end

# Edge morph data: sorted edge x and the prescribed displacement (Δx, Δz) plus
# their x-derivatives (central differences), used to extrapolate into the interior.
function morph_edge_data(grid, nodeset; Ar=80.2/55.2)
    nodes = get_node_coordinate.(getnodes(grid, nodeset))
    idx = sortperm(nodes)
    ns  = nodes[idx]
    x = getindex.(ns, 1);  y = getindex.(ns, 2)
    x_new, y_new = map_initial(x, y, Ar)
    Δx = x_new .- x          # x morph displacement
    Δz = y_new               # z morph displacement (reference z = 0)
    n = length(x)
    dΔx = similar(Δx);  dΔz = similar(Δz)
    for i in 1:n
        lo = max(i-1, 1);  hi = min(i+1, n)
        h  = x[hi] - x[lo]
        dΔx[i] = (Δx[hi] - Δx[lo]) / h
        dΔz[i] = (Δz[hi] - Δz[lo]) / h
    end
    return x, Δx, Δz, dΔx, dΔz
end

# Full-mesh approximate morphed configuration by a transfinite y-blend of the edge
# morph, with directors seeded from the analytic surface normal.  Blend
# φ(y)=½(1+cos(πy/Hy)) carries the edge motion into the interior so the shell starts
# already folded (past the flat-membrane snap) and the fold-induced contraction is
# spread smoothly over the height instead of piling into the corner.
function build_morph_guess(dh, grid; Ar=80.2/55.2)
    coords = get_node_coordinate.(getnodes(grid))
    Hy = maximum(c[2] for c in coords)
    xs, Δx, Δz, dΔx, dΔz = morph_edge_data(grid, "edge"; Ar=Ar)
    blend(y)  =  0.5 * (1 + cos(π * y / Hy))
    blendp(y) = -0.5 * (π / Hy) * sin(π * y / Hy)
    u0 = zeros(ndofs(dh))
    for cell in CellIterator(dh)
        sd = shelldofs(cell)
        X  = getcoordinates(cell)
        for k in 1:length(X)
            x = X[k]
            j = argmin(abs.(xs .- x[1]))           # nearest edge column
            φb = blend(x[2]);  φbp = blendp(x[2])
            ux = φb * Δx[j];   uz = φb * Δz[j]
            # deformed midsurface normal n = ∂ₓp × ∂_y p,  p=(x+φΔx, y, φΔz)
            nx = -φb * dΔz[j]
            ny =  φbp * (φb * dΔz[j] * Δx[j] - (1 + φb * dΔx[j]) * Δz[j])
            nz =  1 + φb * dΔx[j]
            nrm = sqrt(nx^2 + ny^2 + nz^2)
            nx /= nrm;  ny /= nrm;  nz /= nrm
            s = sqrt(nx^2 + ny^2)
            if s < 1e-12
                φ1 = 0.0;  φ2 = 0.0
            else
                α  = acos(clamp(nz, -1.0, 1.0))    # |φ| = tilt from ê_z
                φ1 = nx * α / s;  φ2 = ny * α / s
            end
            b = 5 * (k - 1)
            u0[sd[b+1]] = ux
            u0[sd[b+2]] = 0.0
            u0[sd[b+3]] = uz
            u0[sd[b+4]] = φ1
            u0[sd[b+5]] = φ2
        end
    end
    return u0
end

function assemble_all!(K_int, r_int, dh, scv, u, mat, sdofs, ke, re, u_e)
    asm_i = start_assemble(K_int, r_int)
    for cell in CellIterator(dh)
        sd = sdofs[Ferrite.cellid(cell)]
        reinit!(scv, cell)
        @views u_e .= u[sd]
        fill!(ke, 0.0); fill!(re, 0.0)
        membrane_residuals_RM!(re, scv, u_e, mat)
        bending_residuals_RM!(re, scv, u_e, mat)
        membrane_tangent_RM!(ke, scv, u_e, mat)
        bending_tangent_RM!(ke, scv, u_e, mat)
        assemble!(asm_i, sd, ke, re)
    end
end

# Residual-only assembly (no tangent) for the backtracking line search — the
# expensive MITC/ForwardDiff element tangent is only needed for the Newton
# direction, not for evaluating the residual at a trial step.
function assemble_residual!(r_int, dh, scv, u, mat, sdofs, re, u_e)
    fill!(r_int, 0.0)
    for cell in CellIterator(dh)
        sd = sdofs[Ferrite.cellid(cell)]
        reinit!(scv, cell)
        @views u_e .= u[sd]
        fill!(re, 0.0)
        membrane_residuals_RM!(re, scv, u_e, mat)
        bending_residuals_RM!(re, scv, u_e, mat)
        @views r_int[sd] .+= re
    end
end

function assemble_mass!(M, dh, scv, ρ, mat)
    n_e = ndofs_per_cell(dh)
    me  = zeros(n_e, n_e)
    asm = start_assemble(M)
    for cell in CellIterator(dh)
        fill!(me, 0.0)
        reinit!(scv, cell)
        mass_matrix!(me, scv, ρ, mat)
        assemble!(asm, shelldofs(cell), me)
    end
end

# Follower-pressure load vector + tangent restricted to `cellset` (the
# endocardium SRF_1 ∪ SRF_2 here, not the whole grid).
function assemble_pressure_region!(K_p, F_p, dh, scv, u, cellset, sdofs, ke, re, u_e; Pᵢ=1.0)
    asm = start_assemble(K_p)
    fill!(F_p, 0.0)
    for cell in CellIterator(dh, cellset)
        sd = sdofs[Ferrite.cellid(cell)]
        reinit!(scv, cell)
        @views u_e .= u[sd]
        fill!(ke, 0.0); fill!(re, 0.0)
        assemble_pressure!(re, scv, u_e, Pᵢ)
        assemble_pressure_tangent!(ke, scv, u_e, Pᵢ)
        assemble!(asm, sd, ke)
        @views F_p[sd] .+= re
    end
end

# Pressure residual only (no follower tangent) for the line search.
function assemble_pressure_residual!(F_p, dh, scv, u, cellset, sdofs, re, u_e; Pᵢ=1.0)
    fill!(F_p, 0.0)
    for cell in CellIterator(dh, cellset)
        sd = sdofs[Ferrite.cellid(cell)]
        reinit!(scv, cell)
        @views u_e .= u[sd]
        fill!(re, 0.0)
        assemble_pressure!(re, scv, u_e, Pᵢ)
        @views F_p[sd] .+= re
    end
end
