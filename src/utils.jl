import Ferrite: Grid,Triangle,Quadrilateral,Nodes
using LinearAlgebra: cross
using ForwardDiff

"""
    shell_grid(grid::Grid{2,P,T}; map::Function) where {P<:Union{Triangle,Quadrilateral,QuadraticTriangle,QuadraticQuadrilateral},T}


Embed the 2D `grid` into 3D space by applying the mapping `map` to the nodes (default: flat `z=0`` plane).

For example, the hyperbolic paraboloid shell can be generated in two lines
```julia
# domain ω ∈ ]-1/2; 1/2[ and 3D grid
grid2D = generate_grid(Quadrilateral, (20, 20), Vec(-0.5, -0.5), Vec(0.5, 0.5))
grid3D = shell_grid(grid2D; map=(n)->(n.x[1], n.x[2], n.x[1]^2 - n.x[2]^2))
```
"""
function shell_grid(grid::Grid{2,P,T}; map::Function=(n)->(n.x[1], n.x[2], zero(T))) where {P<:Union{Triangle,Quadrilateral,
                                                                                                     QuadraticTriangle,QuadraticQuadrilateral},T}
    return Grid(grid.cells, [Node(Tensors.Vec{3}(map(n))) for n in grid.nodes];
                facetsets=grid.facetsets, cellsets=grid.cellsets, nodesets=grid.nodesets)
end

import Ferrite: CellCache
"""
    shelldofs(cell)

Reorder DOFs from a two-field `DofHandler` layout (`:u` as ip³, `:θ` as ip²)
to the interleaved 5-DOF-per-node layout expected by the RM assembly functions.

Input layout: ``[u_{1x},u_{1y},u_{1z},\\, u_{2x},\\ldots,u_{nz} \\mid \\theta_{1,1},\\theta_{1,2},\\, \\theta_{2,1},\\ldots,\\theta_{n,2}]``

Output layout: ``[u_{1x},u_{1y},u_{1z},\\theta_{1,1},\\theta_{1,2},\\; u_{2x},u_{2y},u_{2z},\\theta_{2,1},\\theta_{2,2},\\ldots]``

This method reads the layout off the dof count alone, so it is correct only for a
`DofHandler` carrying exactly `:u` then `:θ` and nothing else. Pass the `SubDofHandler`
— `shelldofs(sdh, cell)` or [`shelldofs!`](@ref) — to have the ranges read by name
instead, which stays correct for any field order and any extra fields.
"""
function shelldofs(cell::CellCache)
    dofs = cell.dofs
    rem(length(dofs), 5) == 0 || throw(ArgumentError(
        "shelldofs expects the two-field layout `add!(dh, :u, ip^3); add!(dh, :θ, ip^2)`, " *
        "whose cells carry a multiple of 5 dofs; this cell has $(length(dofs)). " *
        "Use `shelldofs(sdh, cell)` to read the dof ranges by name."))
    n = length(dofs) ÷ 5
    perm = similar(dofs)
    for I in 1:n
        @views perm[5I-4:5I-2] .= dofs[3I-2:3I]
        perm[5I-1] = dofs[3n + 2I-1]
        perm[5I  ] = dofs[3n + 2I]
    end
    return perm
end

"""
    shelldofs(sdh::SubDofHandler, cell) -> Vector{Int}
    shelldofs!(sd::AbstractVector{Int}, sdh::SubDofHandler, cell) -> sd

Layout-safe form of [`shelldofs`](@ref): the same interleaved 5-dof-per-node order,
but built from the `SubDofHandler`'s own `dof_range`s for `:u` and `:θ`, so it stays
correct whatever else the `DofHandler` carries and in whatever order the fields were
added. The in-place form resizes and overwrites `sd` and allocates nothing.
"""
shelldofs(sdh::Ferrite.SubDofHandler, cell) = shelldofs!(Int[], sdh, cell)

"""
    shelldofs!(sd::AbstractVector{Int}, sdh::Ferrite.SubDofHandler, cell) -> sd

In-place, allocation-free form of [`shelldofs`](@ref): resizes and overwrites `sd`
with the interleaved 5-dof-per-node order.
"""
function shelldofs!(sd::AbstractVector{Int}, sdh::Ferrite.SubDofHandler, cell)
    dofs = celldofs(cell)
    ru, rθ = Ferrite.dof_range(sdh, :u), Ferrite.dof_range(sdh, :θ)
    n = length(ru) ÷ 3
    length(ru) == 3n || throw(ArgumentError(":u carries $(length(ru)) dofs per cell, not a multiple of 3"))
    length(rθ) == 2n || throw(ArgumentError(":u spans $n nodes but :θ spans $(length(rθ) / 2)"))
    resize!(sd, 5n)
    @inbounds for I in 1:n
        sd[5I-4] = dofs[ru[3I-2]]
        sd[5I-3] = dofs[ru[3I-1]]
        sd[5I-2] = dofs[ru[3I]]
        sd[5I-1] = dofs[rθ[2I-1]]
        sd[5I  ] = dofs[rθ[2I]]
    end
    sd
end

using OrderedCollections

"""
    get_ferrite_grid(::String; T=Float64)

Loads the `*.inp` file into ferrite an return the `Grid`
"""
# Reverse the winding order of a shell element from CW to CCW (or vice versa).
# Permutation: corners 1,2,3,4 → 1,4,3,2; edge midpoints follow their new corners.
@inline function _flip_shell_nodes!(ns::Vector{Int})
    N = length(ns)
    if N == 3        # Tri3: swap 2↔3
        ns[2], ns[3] = ns[3], ns[2]
    elseif N == 4    # Quad4: swap 2↔4
        ns[2], ns[4] = ns[4], ns[2]
    elseif N == 6    # Tri6: swap 2↔3, 4↔6
        ns[2], ns[3] = ns[3], ns[2]; ns[4], ns[6] = ns[6], ns[4]
    elseif N == 8    # Quad8 (serendipity): swap 2↔4, 5↔8, 6↔7
        ns[2], ns[4] = ns[4], ns[2]; ns[5], ns[8] = ns[8], ns[5]; ns[6], ns[7] = ns[7], ns[6]
    elseif N == 9    # Quad9: swap 2↔4, 5↔8, 6↔7; node 9 (centre) unchanged
        ns[2], ns[4] = ns[4], ns[2]; ns[5], ns[8] = ns[8], ns[5]; ns[6], ns[7] = ns[7], ns[6]
    end
end

function get_ferrite_grid(fname; T=Float64, orient=true)
    #INP file format
    @assert endswith(fname,".inp") "file type not supported"
    fs = open(fname)

    points = Vec{3,T}[]
    faces = Tuple[]
    node_idx = Int[]
    set = 0
    cell_set_list = []
    set_names = String[]

    # read the first 3 lines if there is the "*heading" keyword
    line = readline(fs)
    contains(line,"*heading") && (line = readline(fs))
    BlockType = contains(line,"*NODE") ? Val{:NodeBlock}() : Val{:DataBlock}()

    # read the file
    while !eof(fs)
        line = readline(fs)
        contains(line,"*ELSET, ELSET=") && push!(set_names, split(line,"=")[end])
        (contains(line,"*ELSET, ELSET=") && set>0) && (push!(cell_set_list, set); set=0)
        BlockType, line = parse_blocktype!(BlockType, fs, line)
        if BlockType == Val{:NodeBlock}()
            push!(node_idx, parse(Int,split(line,",")[1])) # keep track of the node index of the inp file
            val = parse.(T,split(line,",")[2:4])
            push!(points, Vec{3,T}(ntuple(i->val[i], 3)))
        elseif BlockType == Val{:ElementBlock}()
            nodes = parse.(Int,split(line,",")[2:end])
            # this returns the index, so it maps to the correct first node
            face_nodes = [findfirst(==(node),node_idx) for node in nodes]
            if orient && length(face_nodes) >= 3
                # Use first two corners and corner 3 (tri) or 4 (quad) to detect winding.
                # If A₁×A₂ points in −z, element is CW → flip to CCW so G₃ = +ê_z.
                x1 = points[face_nodes[1]]; x2 = points[face_nodes[2]]
                x3 = points[face_nodes[length(face_nodes) >= 4 ? 4 : 3]]
                if ((x2-x1) × (x3-x1))[3] < 0
                    _flip_shell_nodes!(face_nodes)
                end
            end
            push!(faces, ntuple(i->face_nodes[i], length(face_nodes))) # parse the face
        elseif BlockType == Val{:ElSetBlock}()
            # push!(set, parse.(Int64,split(line,",")[1]))
            set += 1 # avoid errors when element number is not continuous
        else
            continue
        end
    end
    push!(cell_set_list, set) # don;t forget the last set
    # make the set continuous intervals
    cell_set_list = vcat(0,cumsum(cell_set_list))
    cell_set_list = map(i->cell_set_list[i]+1:cell_set_list[i+1],1:length(cell_set_list)-1)
    close(fs) # close file stream
    # check the lowest node id, must start with 1, otherwise Ferrite breaks
    CellType = get_cell_type(faces)
    grid = Grid(CellType.(faces), Node.(points))
    for (name, set) in zip(set_names, cell_set_list)
        addcellset!(grid, name, Set{Int64}(collect(set)))
    end
    return grid
end
function parse_blocktype!(block, io, line)
    contains(line,"*NODE") && return block=Val{:NodeBlock}(),readline(io)
    contains(line,"*ELEMENT") && return block=Val{:ElementBlock}(),readline(io)
    contains(line,"*ELSET, ELSET=") && return block=Val{:ElSetBlock}(),readline(io)
    return block, line
end
function get_cell_type(faces)
    # Determine the cell type based on the first face
    Nnodes = length(faces[1])
    Nnodes == 3 && return Triangle # S3
    Nnodes == 4 && return Quadrilateral # S4
    Nnodes == 6 && return QuadraticTriangle # S6
    Nnodes == 8 && return SerendipityQuadraticQuadrilateral # S8
    Nnodes == 9 && return QuadraticQuadrilateral # S9
    error("Unsupported cell type")
end

"""
    compute_volume(dh, scv, u; cellset, h, b)

Computes the volume of a shell in the configuration `u`. The default behavior is to use all the `cellset` attached
to the `DofHandler`. By passing unions of cellsets, you can tailor the volume computation to specific regions of the shell.

The vectors ``h`` and ``b`` define the reference and base positions, respectively. These can be used for open shells to remove
contribution to the volume. For example, an inflated membrane on the x-y plane with +z deformation would be measured as
```Julia
vol = compute_volume(dh, scv, u; h=Vec((0.0,0.0,1.0)), b=Vec((0.0,0.0,0.0)))
```
"""
function compute_volume(dh, scv, u::AbstractVector{T}; cellset=1:getncells(dh.grid),
                        h::Vec{3, T}=Vec((0.0,0.0,1.0)), b::Vec{3, T}=Vec((0.0,0.0,0.0))) where T
    volume = zero(T)
    for cell in CellIterator(dh, cellset)
        reinit!(scv, cell)
        coords = getcoordinates(cell)
        uₑ = u[shelldofs(cell)] # arranged as [u₁,u₂,u₃,φ₁,φ₂,…]
        volume += volume_residual(scv, coords, uₑ, h, b)
    end
    return volume
end

function volume_residual(scv, coords, uₑ::AbstractVector{T}, h, b) where T
    val = zero(T)
    for qp in 1:getnquadpoints(scv)
        d = function_value(scv, qp, uₑ)
        n = getnormal(scv, qp)
        x = spatial_coordinate(scv, qp, coords)
        ∇u = function_gradient(scv, qp, uₑ)
        F = one(∇u) + ∇u
        val +=  det(F) * ((h ⊗ h) ⋅ (x + d - b)) ⋅ (transpose(inv(F)) ⋅ n) * getdetJdV(scv, qp)
    end
    return -val
end

"""
    volume_residuals!(re, dh, scv, u, V⁰; cellset, h, b)

Compute the volume residuals ``r =  V^0 - \\oint J(\\vec{h}\\otimes\\vec{h}) \\cdot (\\vec{x} + \\vec{d} - \\vec{b} ) \\cdot  (F^{-\\top}\\cdot\\vec{n}) \\text{ d}\\Omega``.
The residual is stored in the first index of the `re` vector.

The default behavior is to use all the `cellset` attached to the `DofHandler`. By passing unions of cellsets, you can tailor the volume computation to specific regions of the shell.

See also [`compute_volume`](@ref).
"""
function volume_residuals!(re, dh, scv::ShellCellValues, u::AbstractVector{T}, V⁰; cellset=1:getncells(dh.grid),
                           h::Vec{3,T}=Vec((0.0,0.0,1.0)), b::Vec{3,T}=Vec((0.0,0.0,0.0))) where T
    for cell in CellIterator(dh, cellset)
        reinit!(scv, cell)
        coords = getcoordinates(cell)
        uₑ = u[shelldofs(cell)]
        re[1] += volume_residual(scv, coords, uₑ, h, b)
    end
    re[1] += V⁰
end

"""
    volume_gradient!(dVdu, dh, scv, u; cellset, h, b)

Compute the volume gradient ``\\partial V_{3D}/\\partial u`` into `dVdu` via ForwardDiff.
Each element contribution is `ForwardDiff.gradient(ue -> volume_residual(..., ue, h, b), ue)`
assembled into the global DOF vector using the shell DOF permutation.

See also [`compute_volume`](@ref).
"""
function volume_gradient!(dVdu, dh, scv::ShellCellValues, u::AbstractVector{T}; cellset=1:getncells(dh.grid),
                          h::Vec{3,T}=Vec((0.0,0.0,1.0)), b::Vec{3,T}=Vec((0.0,0.0,0.0))) where T
    fill!(dVdu, zero(T))
    for cell in CellIterator(dh, cellset)
        reinit!(scv, cell)
        coords = getcoordinates(cell)
        sd  = shelldofs(cell)
        uₑ  = u[sd]
        #TODO this could be replaced by an expression to save allocations
        dVdu[sd] .+= ForwardDiff.gradient(v -> volume_residual(scv, coords, v, h, b), uₑ)
    end
end

"""
    director_field(dh, scv, u) -> (d, G3)

Compute per-node deformed director `d` and reference shell normal `G3` from the
displacement/rotation solution `u`. Both are returned as `3 × n_nodes` matrices.

Each nodal value is the element-average of the QP-level frame vectors, accumulated
and averaged over all elements sharing the node.

The director is computed from the Rodrigues rotation formula

```math
d_I = \\cos|\\varphi|\\, G_3 + \\operatorname{sinc}|\\varphi|\\,(\\varphi_1 T_1 + \\varphi_2 T_2)
```

which preserves unit length exactly for any rotation magnitude.
Requires a two-field `DofHandler` with `:u` (ip³) and `:θ` (ip²).

# Example
```julia
d, G3 = director_field(dh, scv, u)
VTKGridFile("output", dh) do vtk
    write_solution(vtk, dh, u)
    Ferrite.write_node_data(vtk, d,  "director")
    Ferrite.write_node_data(vtk, G3, "G3")
end
```
"""
function director_field(dh::DofHandler, scv::ShellCellValues, u)
    n_nodes = getnnodes(dh.grid)
    d_sum  = zeros(3, n_nodes)
    G3_sum = zeros(3, n_nodes)
    count  = zeros(Int, n_nodes)
    for cell in CellIterator(dh)
        reinit!(scv, cell)
        sd  = shelldofs(cell)
        u_e = @views u[sd]
        nq  = getnquadpoints(scv)
        G3_avg = sum(scv.G₃[q] for q in 1:nq) / nq
        T1_avg = sum(scv.T₁[q] for q in 1:nq) / nq
        T2_avg = sum(scv.T₂[q] for q in 1:nq) / nq
        for (I, nid) in enumerate(cell.nodes)
            φ₁ = u_e[5I-1]; φ₂ = u_e[5I]
            cosθ, sincθ = cos_sinc_sq(φ₁^2 + φ₂^2)
            d_I = cosθ * G3_avg + sincθ * (φ₁ * T1_avg + φ₂ * T2_avg)
            @views d_sum[:, nid]  .+= d_I
            @views G3_sum[:, nid] .+= G3_avg
            count[nid] += 1
        end
    end
    for i in 1:n_nodes
        c = count[i]
        if c > 0
            @views d_sum[:, i]  ./= c
            @views G3_sum[:, i] ./= c
        end
    end
    return d_sum, G3_sum
end

"""
    shell_strains(scv, qp, u_e) -> (E, κ, γ)

Compute all three RM shell strain measures at quadrature point `qp` from a flat
5-DOF/node element vector `u_e = [u₁,u₂,u₃,φ₁,φ₂, …]`.

Returns:
- `E :: SymmetricTensor{2,2}` — membrane strain, Green–Lagrange: Eαβ = ½(aα·aβ − Aα·Aβ)
- `κ :: SymmetricTensor{2,2}` — bending curvature change: καβ = ½(aα·d,β + aβ·d,α) − Bαβ
- `γ :: Vec{2}` — transverse shear strain: γα = aα·d − Aα·G₃ (MITC-corrected if applicable)
"""
function shell_strains(scv::ShellCellValues, qp::Int, u_e::AbstractVector{T}) where T
    n_nodes = getnbasefunctions(scv.ip_shape)

    Δa₁ = zero(Vec{3,T}); Δa₂ = zero(Vec{3,T})
    for I in 1:n_nodes
        u_I  = Vec{3,T}((u_e[5I-4], u_e[5I-3], u_e[5I-2]))
        Δa₁ += u_I * scv.dNdξ[I, qp][1]
        Δa₂ += u_I * scv.dNdξ[I, qp][2]
    end
    a₁ = scv.A₁[qp] + Δa₁
    a₂ = scv.A₂[qp] + Δa₂

    E = membrane_strain(a₁, a₂, scv.A_metric[qp])

    d, d₁, d₂ = director_field(scv, qp, u_e, n_nodes)

    κ = curvature_tensor(a₁, a₂, d₁, d₂, scv.B₀[qp])

    γ_k = tying_shear_strains(scv.mitc, u_e)
    γ₁, γ₂ = shear_strains(a₁, a₂, d, qp, γ_k, scv.mitc)
    d₀  = reference_director(scv, qp, n_nodes)
    r₁, r₂ = reference_shear_offset(scv.A₁[qp], scv.A₂[qp], d₀, scv.mitc)
    γ₁ -= r₁; γ₂ -= r₂

    return E, κ, Vec{2,T}((γ₁, γ₂))
end

"""
    embed23(S) -> SymmetricTensor{2,3}

Embed a surface `SymmetricTensor{2,2}` into a 3D symmetric tensor by padding
the out-of-plane rows/columns with zeros. Useful for writing shell strain or
stress tensors to VTK (ParaView expects 6-component symmetric tensors).
"""
@inline embed23(S::SymmetricTensor{2,2,T}) where T = SymmetricTensor{2,3,T}((S[1,1], S[1,2], zero(T), S[2,2], zero(T), zero(T)))

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
reinit!

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
            for (I, nid) in enumerate(getnodes(cell))
                dofmap[nid] = (cd[rθ[2I-1]], cd[rθ[2I]])
            end
        end
    end
    dofmap
end

