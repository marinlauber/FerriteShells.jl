using FerriteShells
using OrderedCollections

"""
    Read inp file into a...
"""
function load_inp(fname; T=Float64)
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

# test it on a mesh
# L = 1.0
# n = 4
# corners = [Vec{2}((0.0, 0.0)), Vec{2}((L/2, 0.0)), Vec{2}((L/2, L/2)), Vec{2}((0.0, L/2))]
# grid0 = shell_grid(generate_grid(Quadrilateral, (n, n), corners))
# addfacetset!(grid0, "facet1", x -> x[1] ≈ 0.0)
# addcellset!(grid0, "all", x->true)

# # open a sample file
# fname = "/home/marin/Workspace/HHH/code/square_pillow/square_pillow_N61_S3/geom.inp"
# fname = "/home/marin/Workspace/WaterLilyPreCICE/meshes/cube.inp"
fname = "/home/marin/Workspace/HHH/code/limo/p6/geom.inp"

grid = load_inp(fname)

# ip = Lagrange{RefTriangle, 1}()
ip = Serendipity{RefQuadrilateral, 2}()

dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

u = zeros(ndofs(dh))

VTKGridFile("test_read", dh) do vtk
    write_solution(vtk, dh, u)
    Ferrite.write_cellset(vtk, grid)
end
