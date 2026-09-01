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
    reinit_geometry!(scv, x)
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

"""
    add_director_symmetry!(ch::ConstraintHandler, dh::DofHandler, nf::NodeFrames,
                           nodeset_name::String, n::Vec{3}; atol=0.25)

Constrain the director to stay in the symmetry plane with unit normal `n` at every node
of `nodeset_name`, i.e. ``\\mathbf{d}\\cdot\\mathbf{n} = 0``.

With the Rodrigues director ``\\mathbf{d} = \\cos\\theta\\,\\mathbf{G}_3 +
\\mathrm{sinc}\\,\\theta\\,(\\varphi_1\\mathbf{T}_1 + \\varphi_2\\mathbf{T}_2)`` and
``\\mathbf{G}_3\\cdot\\mathbf{n} = 0`` — which holds exactly on a symmetry plane — this is
the *exact* linear constraint

```math
\\varphi_1 (\\mathbf{T}_1\\cdot\\mathbf{n}) + \\varphi_2 (\\mathbf{T}_2\\cdot\\mathbf{n}) = 0
```

added as a Ferrite `AffineConstraint` on the better-conditioned of the two components.

**Why not just fix ``\\varphi_2``:** ``\\varphi_1,\\varphi_2`` are components in the nodal
frame, and that frame is built from `G₃` by a heuristic (`ref = |G₃_x| < 0.9 ? ê_x : ê_y`)
that *flips* as the normal sweeps past the threshold. `Dirichlet(:θ, set, x -> 0.0, [2])`
therefore means different physical constraints on different parts of the same boundary —
on a hemisphere the frame flips right at the equator and the constraint silently becomes
a spurious clamp. No continuous tangent frame exists on a closed curved surface, so this
cannot be fixed by a better heuristic; the constraint has to be written frame-independently,
which is what this function does.

`nf` **must be the same `NodeFrames` the assembly `reinit!`s with** — the constraint is
expressed in the nodal frame, so with per-element (centroid) frames the `φ` DOFs at a node
have no single meaning and the constraint is ill-posed.

A node lying on two symmetry planes needs ``\\mathbf{d} = \\mathbf{G}_3``, i.e.
``\\varphi_1 = \\varphi_2 = 0``; add that as an ordinary `Dirichlet` on `:θ` and keep the
node out of the sets passed here.
"""
function add_director_symmetry!(ch::ConstraintHandler, dh::DofHandler, nf::NodeFrames,
                                nodeset_name::String, n::Vec{3}; atol = 0.25)
    norm(n) > 0 || throw(ArgumentError("n must be nonzero"))
    n̂ = n / norm(n)
    dofmap = _theta_dofmap(dh)
    for nid in sort!(collect(getnodeset(dh.grid, nodeset_name)))
        haskey(dofmap, nid) ||
            throw(ArgumentError("node $nid carries no :θ field; is it in the shell subdomain?"))
        G₃ = nf.G₃[nid]
        abs(G₃ ⋅ n̂) ≤ atol || throw(ArgumentError(
            "node $nid: G₃·n = $(round(G₃ ⋅ n̂; digits=4)) exceeds atol = $atol, so `n` is not a " *
            "symmetry-plane normal there (a symmetry plane contains the shell normal). Note the " *
            "nodal normal is a one-sided area average on a boundary and tilts O(h) out of the " *
            "plane, so a modest tilt is expected here; raise `atol` to accept more."))
        a = nf.T₁[nid] ⋅ n̂
        b = nf.T₂[nid] ⋅ n̂
        d₁, d₂ = dofmap[nid]
        # (T₁,T₂,G₃) is orthonormal, so a² + b² = 1 - (G₃·n̂)² ≥ 1 - atol² (enforced
        # above, atol defaults to 0.25): never both small.
        if abs(a) ≥ abs(b)
            add!(ch, Ferrite.AffineConstraint(d₁, [d₂ => -b / a], 0.0))
        else
            add!(ch, Ferrite.AffineConstraint(d₂, [d₁ => -a / b], 0.0))
        end
    end
    return ch
end

# node id -> (φ₁ dof, φ₂ dof), read by name so any field order/extra fields are fine
function _theta_dofmap(dh::DofHandler)
    dofmap = Dict{Int, Tuple{Int, Int}}()
    for sdh in dh.subdofhandlers
        :θ in Ferrite.getfieldnames(sdh) || continue
        rθ = Ferrite.dof_range(sdh, :θ)
        for cell in CellIterator(sdh)
            cd = celldofs(cell)
            nn = length(getnodes(cell))
            length(rθ) == 2nn || throw(ArgumentError("expected :θ to have 2 DOFs per node (φ₁, φ₂); got $(length(rθ)) DOFs per cell for $nn nodes"))
            for (I, nid) in enumerate(getnodes(cell))
                dofmap[nid] = (cd[rθ[2I-1]], cd[rθ[2I]])
            end
        end
    end
    dofmap
end

