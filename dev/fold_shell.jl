using FerriteShells
using LinearAlgebra

# Two flat Q4 RM plates joined at a fold line via AffineConstraints.
#
# Geometry: plate 1 in x-y plane (z=0), plate 2 in x-z plane (y=0).
# Fold line at y=0, z=0 (the x-axis). Both plates are clamped at their far edge.
# A uniform follower pressure is applied to plate 1 only.
#
# Fold-line nodes are duplicated: plate1 fold nodes ↔ plate2 fold nodes.
# AffineConstraints tie u DOFs (displacement continuity) and θ₁ DOFs (rotation
# about the fold axis êₓ) across the fold → rigid seam, not a hinge.
# θ₂ is left free on each side: it rotates about T₂=êᵧ (plate 1) vs T₂=êᵤ
# (plate 2), different physical axes that don't couple at a 90° fold.

const L  = 0.1     # plate side length [m]
const T  = 0.002   # thickness [m]
const E  = 70e9    # Young's modulus [Pa]
const NU = 0.3     # Poisson ratio
const P0 = 5e3     # pressure on plate 1 [Pa]

function make_fold_grid(n)
    node1(i, j) = j*(n+1) + i + 1                  # plate 1: x-y plane
    node2(i, j) = (n+1)^2 + j*(n+1) + i + 1        # plate 2: x-z plane

    nodes = Node{3, Float64}[]
    for j in 0:n, i in 0:n
        push!(nodes, Node(Vec{3}((i*L/n, j*L/n, 0.0))))
    end
    for j in 0:n, i in 0:n
        push!(nodes, Node(Vec{3}((i*L/n, 0.0, j*L/n))))
    end

    cells = Quadrilateral[]
    for j in 0:n-1, i in 0:n-1
        push!(cells, Quadrilateral((node1(i,j), node1(i+1,j), node1(i+1,j+1), node1(i,j+1))))
    end
    for j in 0:n-1, i in 0:n-1
        push!(cells, Quadrilateral((node2(i,j), node2(i+1,j), node2(i+1,j+1), node2(i,j+1))))
    end

    grid = Grid(cells, nodes)
    addcellset!(grid, "plate1", Set{Int}(1:n^2))
    addcellset!(grid, "plate2", Set{Int}(n^2+1:2n^2))
    addnodeset!(grid, "clamp1", Set{Int}(node1(i, n) for i in 0:n))
    addnodeset!(grid, "clamp2", Set{Int}(node2(i, n) for i in 0:n))

    fold_pairs = [node1(i, 0) => node2(i, 0) for i in 0:n]
    return grid, fold_pairs
end

n   = 8
grid, fold_pairs = make_fold_grid(n)

ip  = Lagrange{RefQuadrilateral, 1}()
qr  = QuadratureRule{RefQuadrilateral}(2)
scv = ShellCellValues(qr, ip, ip)

dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

# Build node → raw u-DOF and θ-DOF indices (DH ordering, not shelldofs).
# For a two-field DH (:u ip^3, :θ ip^2) with n_loc nodes per cell:
#   u DOFs for node I: cd[3I-2:3I]
#   θ₁ DOF for node I: cd[3*n_loc + 2I-1]
#   θ₂ DOF for node I: cd[3*n_loc + 2I]
n_nodes    = getnnodes(grid)
node_udofs = [Int[] for _ in 1:n_nodes]
node_θ1dof = zeros(Int, n_nodes)
for cell in CellIterator(dh)
    cd    = celldofs(cell)
    n_loc = length(cell.nodes)
    for I in 1:n_loc
        nid = cell.nodes[I]
        if isempty(node_udofs[nid])
            append!(node_udofs[nid], @views cd[3I-2:3I])
            node_θ1dof[nid] = cd[3n_loc + 2I-1]
        end
    end
end

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getnodeset(grid, "clamp1"), x -> zeros(3), [1,2,3]))
add!(ch, Dirichlet(:θ, getnodeset(grid, "clamp1"), x -> zeros(2), [1,2]))
add!(ch, Dirichlet(:u, getnodeset(grid, "clamp2"), x -> zeros(3), [1,2,3]))
add!(ch, Dirichlet(:θ, getnodeset(grid, "clamp2"), x -> zeros(2), [1,2]))

# Rigid fold: tie u DOFs and θ₁ (rotation about the fold axis êₓ, T₁ on both plates).
# θ₂ is left free: it represents rotation about T₂=êᵧ (plate 1) vs T₂=êᵤ (plate 2),
# which are different physical axes and don't couple at a 90° fold.
for (master, slave) in fold_pairs
    for (dm, ds) in zip(node_udofs[master], node_udofs[slave])
        add!(ch, AffineConstraint(ds, [dm => 1.0], 0.0))
    end
    add!(ch, AffineConstraint(node_θ1dof[slave], [node_θ1dof[master] => 1.0], 0.0))
end

close!(ch)
update!(ch, 0.0)

mat = LinearElastic(E, NU, T)

K = allocate_matrix(dh)
f = zeros(ndofs(dh))
assembler = start_assemble(K, f)

n_e = ndofs_per_cell(dh)
ke  = zeros(n_e, n_e)
fe  = zeros(n_e)
plate1 = getcellset(grid, "plate1")

for cell in CellIterator(dh)
    fill!(ke, 0.0); fill!(fe, 0.0)
    reinit!(scv, cell)
    sd  = shelldofs(cell)
    u_e = zeros(n_e)
    membrane_tangent_RM!(ke, scv, u_e, mat)
    bending_tangent_RM!(ke, scv, u_e, mat)
    cellid(cell) in plate1 && assemble_pressure!(fe, scv, u_e, -P0)
    assemble!(assembler, sd, ke, fe)
end

apply!(K, f, ch)
u = K \ f
apply!(u, ch)

VTKGridFile("fold_shell", dh) do vtk
    write_solution(vtk, dh, u)
    write_directors!(vtk, dh, scv, u)
end

uz_plate1 = [u[node_udofs[j*(n+1)+i+1][3]] for i in 0:n, j in 0:n]
println("Max |u_z| on plate 1 = $(maximum(abs.(uz_plate1))) m")