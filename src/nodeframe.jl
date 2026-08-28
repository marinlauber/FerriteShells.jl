"""
    NodeFrames

Per-node area-weighted averaged director frames for a shell mesh. Eliminates the
O(h/R) inter-element frame inconsistency that occurs when adjacent curved-shell
elements each derive their own centroid frame.

Construct via `NodeFrames(grid, ip_geo)`. Pass to `reinit!(scv, x, nf, node_ids)`
instead of the plain `reinit!(scv, x)` to activate per-node frames.

For flat shells the result is identical to the centroid-frame approach.
"""
struct NodeFrames
    G₃ :: Vector{Vec{3,Float64}}
    T₁ :: Vector{Vec{3,Float64}}
    T₂ :: Vector{Vec{3,Float64}}
end

function NodeFrames(grid::Grid, ip_geo::Interpolation)
    n_nodes = getnnodes(grid)
    G₃_sum  = fill(zero(Vec{3,Float64}), n_nodes)

    for cellid in 1:getncells(grid)
        cell     = getcells(grid, cellid)
        node_ids = collect(cell.nodes)
        x        = [grid.nodes[nid].x for nid in node_ids]
        ξ_c      = reference_centroid(ip_geo)
        A₁ = zero(Vec{3,Float64}); A₂ = zero(Vec{3,Float64})
        n_geo = getnbasefunctions(ip_geo)
        for i in 1:n_geo
            dN, _ = Ferrite.reference_shape_gradient_and_value(ip_geo, ξ_c, i)
            A₁ += x[i] * dN[1]; A₂ += x[i] * dN[2]
        end
        n_vec = A₁ × A₂
        area  = norm(n_vec)
        G₃_c  = n_vec / area
        for nid in node_ids
            G₃_sum[nid] += area * G₃_c
        end
    end

    G₃ = Vector{Vec{3,Float64}}(undef, n_nodes)
    T₁ = Vector{Vec{3,Float64}}(undef, n_nodes)
    T₂ = Vector{Vec{3,Float64}}(undef, n_nodes)
    for i in 1:n_nodes
        g      = G₃_sum[i]
        g_norm = norm(g)
        g_norm < 1e-14 && continue
        G₃[i] = g / g_norm
        ref    = abs(G₃[i][1]) < 0.9 ? Vec{3}((1.,0.,0.)) : Vec{3}((0.,1.,0.))
        t₁     = ref - (ref ⋅ G₃[i]) * G₃[i]
        T₁[i]  = t₁ / norm(t₁)
        T₂[i]  = G₃[i] × T₁[i]
    end
    NodeFrames(G₃, T₁, T₂)
end

"""
    reinit!(scv::ShellCellValues, x::AbstractVector, nf::NodeFrames)
    reinit!(scv::ShellCellValues, cc::CellCache, nf::NodeFrames)
    reinit!(scv::ShellCellValues, cell::AbstractCell, nf::NodeFrames)

Update the `ShellCellValues` object for a cell with cell coordinates `x` and a `NodeFrames` object.

The reference surface measures such as the covariant basis are obtained from the `NodeFrames` object
pre-computed initially from `nf = NodeFrames(grid, ip_geo)`.

**Note:**
For `ShellCellValues` where a shear treatment has been specified, the `MITC` data is also `reinit!`.
"""
reinit!(::ShellCellValues, x::AbstractVector{<:Vec{3}}, ::NodeFrames, node_ids)

reinit!(scv::ShellCellValues, cell, nf::NodeFrames) = reinit!(scv, getcoordinates(cell), nf, getnodes(cell))
reinit!(scv::ShellCellValues, cc::CellCache, nf::NodeFrames) = reinit!(scv, getcoordinates(cc), nf, getnodes(cc))
function reinit!(scv::ShellCellValues, x::AbstractVector{<:Vec{3}}, nf::NodeFrames, node_ids)
    reinit!(scv, x)
    # The frames live on `ip_shape` nodes (that is how `G₃_elem` is sized and how
    # `reference_director_curvature!` reads them), but `nf` is indexed by grid node id.
    # The two only line up when every shape node is a grid node.
    n_shape = getnbasefunctions(scv.ip_shape)
    length(node_ids) ≥ n_shape || throw(ArgumentError(
        "NodeFrames needs one frame per shape node, but the cell carries $(length(node_ids)) " *
        "nodes for $n_shape shape functions. Use an `ip_shape` whose nodes are grid nodes " *
        "(e.g. `ip_geo == ip_shape`), or call `reinit!(scv, x)` for centroid frames."))
    for I in 1:n_shape
        scv.G₃_elem[I] = nf.G₃[node_ids[I]]
        scv.T₁_elem[I] = nf.T₁[node_ids[I]]
        scv.T₂_elem[I] = nf.T₂[node_ids[I]]
    end
    reference_director_curvature!(scv)   # B₀ follows the frames actually in use
    reinit!(scv.mitc, scv.ip_geo, x, scv.G₃_elem, scv.T₁_elem, scv.T₂_elem)
end
