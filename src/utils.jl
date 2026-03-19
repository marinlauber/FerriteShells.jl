import Ferrite: Grid,Triangle,Quadrilateral,Nodes
using LinearAlgebra: cross
using ForwardDiff

# maps the 2D nodes of a mesh onto the 3D coordinates
# by applying the `map` function to the nodes (default: flat z=0 plane)
function shell_grid(grid::Grid{2,P,T}; map::Function=(n)->(n.x[1], n.x[2], zero(T))) where {P<:Union{Triangle,Quadrilateral,
                                                                                                     QuadraticTriangle,QuadraticQuadrilateral},T}
    return Grid(grid.cells, [Node(Tensors.Vec{3}(map(n))) for n in grid.nodes])
end

function compute_membrane_strains(Es, scv, u_e)
    for qp in 1:getnquadpoints(scv)
        _, _, A_metric, a_metric = kinematics(scv, qp, u_e)
        Es[qp] = 0.5 * (a_metric - A_metric)
    end
end


# K.J - Bath https://doi.org/10.1016/S0045-7949(03)00010-5
function s_norm(u, uₕ)
    nothing
end


import Ferrite: CellCache
"""
    shelldofs()

Reorder DOFs from a two-field DofHandler layout (:u as ip^3, :θ as ip^2)
to the interleaved 5-DOF-per-node layout expected by the RM assembly functions.
Input:  [u1x,u1y,u1z, u2x,...,unz | θ1₁,θ1₂, θ2₁,...,θn₂]
Output: [u1x,u1y,u1z,θ1₁,θ1₂, u2x,u2y,u2z,θ2₁,θ2₂, ...]
"""
function shelldofs(cell::CellCache)
    dofs = cell.dofs
    n = length(dofs) ÷ 5
    perm = similar(dofs)
    for I in 1:n
        @views perm[5I-4:5I-2] .= dofs[3I-2:3I]
        perm[5I-1] = dofs[3n + 2I-1]
        perm[5I  ] = dofs[3n + 2I]
    end
    return perm
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

function compute_volume(dh, scv, u::AbstractVector{T}) where T
    volume = 0.0
    h::Vec{3, T} = Vec((0.0, 0.0, 1.0))
    b::Vec{3, T} = Vec((0.0, 0.0, -0.1))
    for cell in CellIterator(dh)
        reinit!(scv, cell)
        coords = getcoordinates(cell)
        uₑ = u[shelldofs(cell)] # arranged as [u₁,u₂,u₃,φ₁,φ₂,…]
        for qp in 1:getnquadpoints(scv)
            d = function_value(scv, qp, uₑ)
            n = getnormal(scv, qp)
            x = spatial_coordinate(scv, qp, coords)
            ∇u = function_gradient(scv, qp, uₑ)
            F = one(∇u) + ∇u
            volume -=  det(F) * ((h ⊗ h) ⋅ (x + d - b)) ⋅ (transpose(inv(F)) ⋅ n) * getdetJdV(scv, qp)
        end
    end
    return volume
end