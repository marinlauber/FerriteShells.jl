using FerriteShells, LinearAlgebra

# Y-shell: all surfaces start flat (z=0); top arm tip is pulled up by L/10 via prescribed BC.
# Side view:   ──────+──── top arm ↑ (tip pulled to z=L/10)
#                    |
#              ──────+──── bot arm (free)
# Shared nodes at the junction (x=L) belong to stem + both arm elements.

function make_y_shell_grid(; nx_stem=4, nx_arm=4, ny=4, L=1.0, W=0.5)
    nn = ny + 1

    nodes = Vec{3,Float64}[]
    for i in 0:nx_stem, j in 0:ny
        push!(nodes, Vec{3}((i*L/nx_stem, j*W/ny, 0.0)))
    end
    for i in 1:nx_arm, j in 0:ny
        push!(nodes, Vec{3}((L + i*L/nx_arm, j*W/ny, 0.0)))
    end
    for i in 1:nx_arm, j in 0:ny
        push!(nodes, Vec{3}((L + i*L/nx_arm, j*W/ny, 0.0)))
    end

    stem_node(i, j) = i*nn + j + 1
    top_node(i, j)  = i == 0 ? stem_node(nx_stem, j) : (nx_stem+1)*nn + (i-1)*nn + j + 1
    bot_node(i, j)  = i == 0 ? stem_node(nx_stem, j) : (nx_stem+1)*nn + nx_arm*nn + (i-1)*nn + j + 1

    cells = Quadrilateral[]
    for i in 0:nx_stem-1, j in 0:ny-1
        push!(cells, Quadrilateral((stem_node(i,j), stem_node(i+1,j), stem_node(i+1,j+1), stem_node(i,j+1))))
    end
    for i in 0:nx_arm-1, j in 0:ny-1
        push!(cells, Quadrilateral((top_node(i,j), top_node(i+1,j), top_node(i+1,j+1), top_node(i,j+1))))
    end
    for i in 0:nx_arm-1, j in 0:ny-1
        push!(cells, Quadrilateral((bot_node(i,j), bot_node(i+1,j), bot_node(i+1,j+1), bot_node(i,j+1))))
    end

    n_stem = nx_stem * ny
    n_arm  = nx_arm  * ny
    grid = Grid(cells, Node.(nodes))
    addcellset!(grid, "stem",    1:n_stem)
    addcellset!(grid, "top_arm", n_stem+1:n_stem+n_arm)
    addcellset!(grid, "bot_arm", n_stem+n_arm+1:n_stem+2n_arm)
    addnodeset!(grid, "left",      x -> x[1] ≈ 0.0)
    # top_right: built from connectivity since both arm tips share the same x=2L coordinates
    addnodeset!(grid, "top_right", Set(top_node(nx_arm, j) for j in 0:ny))
    addnodeset!(grid, "bottom_right", Set(bot_node(nx_arm, j) for j in 0:ny))
    return grid
end

function assemble_y!(K, f, dh, scv, u, mat)
    stem_set = getcellset(dh.grid, "stem")
    top_set  = getcellset(dh.grid, "top_arm")
    n = ndofs_per_cell(dh)
    ke = zeros(n, n); re = zeros(n)
    asm = start_assemble(K, f)
    for cell in CellIterator(dh)
        fill!(ke, 0.0); fill!(re, 0.0)
        reinit!(scv, cell)
        id  = Ferrite.cellid(cell)
        u_e = u[shelldofs(cell)]
        membrane_tangent_RM!(ke, scv, u_e, mat)
        bending_tangent_RM!(ke, scv, u_e, mat)
        membrane_residuals_RM!(re, scv, u_e, mat)
        bending_residuals_RM!(re, scv, u_e, mat)
        assemble!(asm, shelldofs(cell), ke, re)
    end
end

L = 1.0; W = 0.5
grid = make_y_shell_grid(nx_stem=8, nx_arm=8, ny=4, L=L, W=W)

ip  = Lagrange{RefQuadrilateral,1}()
qr  = QuadratureRule{RefQuadrilateral}(2)
scv = ShellCellValues(qr, ip, ip; mitc=MITC4)

dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

mat = LinearElastic(200e3, 0.3, 0.005)

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getnodeset(grid, "left"),      x -> zeros(3), [1,2,3]))
add!(ch, Dirichlet(:θ, getnodeset(grid, "left"),      x -> zeros(2), [1,2]))
add!(ch, Dirichlet(:u, getnodeset(grid, "top_right"), x -> L/5,     [3]))
add!(ch, Dirichlet(:u, getnodeset(grid, "bottom_right"), x -> 0.0,     [3]))
close!(ch); update!(ch, 0.0)

K = allocate_matrix(dh)
f = zeros(ndofs(dh))
u = zeros(ndofs(dh))

assemble_y!(K, f, dh, scv, u, mat)
apply!(K, f, ch)
u .= K \ f

@show maximum(abs.(u[3:5:end]))  # max z-displacement

VTKGridFile("y_shell_doublelayer", dh) do vtk
    write_solution(vtk, dh, u)
end
