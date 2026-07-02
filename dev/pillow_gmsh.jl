using FerriteShells, LinearAlgebra, Printf, WriteVTK
using Gmsh, FerriteGmsh

"""
    make_pillow_gmsh_grid(; L=1.0, n=16)

Quarter-pillow grid [0,L/2]² using Gmsh unstructured recombined Q9 elements.
`n` controls target element count per side (mesh size h = L/2/n).
Physical groups → Ferrite facetsets: "edge", "sym_x", "sym_y".
"""
function make_pillow_gmsh_grid(; L=1.0, n=16)
    h = (L/2) / n
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)  # Q9, not Q8
    gmsh.model.add("pillow")

    p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, h)
    p2 = gmsh.model.geo.addPoint(L/2, 0.0, 0.0, h)
    p3 = gmsh.model.geo.addPoint(L/2, L/2, 0.0, h)
    p4 = gmsh.model.geo.addPoint(0.0, L/2, 0.0, h)

    l_sym_y  = gmsh.model.geo.addLine(p1, p2)
    l_edge_x = gmsh.model.geo.addLine(p2, p3)
    l_edge_y = gmsh.model.geo.addLine(p3, p4)
    l_sym_x  = gmsh.model.geo.addLine(p4, p1)

    cl = gmsh.model.geo.addCurveLoop([l_sym_y, l_edge_x, l_edge_y, l_sym_x])
    s  = gmsh.model.geo.addPlaneSurface([cl])

    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(1, [l_edge_x, l_edge_y], -1, "edge")
    gmsh.model.addPhysicalGroup(1, [l_sym_x],            -1, "sym_x")
    gmsh.model.addPhysicalGroup(1, [l_sym_y],            -1, "sym_y")
    gmsh.model.addPhysicalGroup(2, [s],                  -1, "surface")

    gmsh.model.mesh.setRecombine(2, s)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)

    grid2d = togrid()
    gmsh.finalize()

    grid = shell_grid(grid2d)
    addnodeset!(grid, "center", x -> norm(x) < 1e-10)
    return grid
end

function make_quarter_pillow_grid(n; L=1.0, primitive=QuadraticQuadrilateral2)
    corners = [Vec{2}((0.0, 0.0)), Vec{2}((L/2, 0.0)), Vec{2}((L/2, L/2)), Vec{2}((0.0, L/2))]
    grid2d = generate_grid(primitive, (n, n), corners)
    grid = shell_grid(grid2d)
    return grid
end

function static_solve(grid, mat, p_max; max_iter=20, tol=1e-8)
    ip  = Lagrange{RefQuadrilateral, 2}()
    qr  = QuadratureRule{RefQuadrilateral}(3)
    scv = ShellCellValues(qr, ip, ip; mitc=MITC9)

    dh = DofHandler(grid)
    add!(dh, :u, ip^3)
    add!(dh, :θ, ip^2)
    close!(dh)

    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "edge"),  x -> 0.0,      [3]))
    add!(ch, Dirichlet(:θ, getfacetset(grid, "edge"),  x -> zeros(2), [1,2]))
    add!(ch, Dirichlet(:u, getfacetset(grid, "sym_x"), x -> 0.0,      [1]))
    add!(ch, Dirichlet(:θ, getfacetset(grid, "sym_x"), x -> 0.0,      [1]))
    add!(ch, Dirichlet(:u, getfacetset(grid, "sym_y"), x -> 0.0,      [2]))
    add!(ch, Dirichlet(:θ, getfacetset(grid, "sym_y"), x -> 0.0,      [2]))
    close!(ch); Ferrite.update!(ch, 0.0)

    N_dof = ndofs(dh)
    free  = ch.free_dofs
    K_int = allocate_matrix(dh)
    K_p   = allocate_matrix(dh)
    r_int = zeros(N_dof)
    F_p   = zeros(N_dof)
    n_e   = ndofs_per_cell(dh)
    ke    = zeros(n_e, n_e); re = zeros(n_e)

    u = zeros(N_dof); apply!(u, ch)

    for iter in 1:max_iter
        fill!(r_int, 0.0); fill!(K_int, 0.0); fill!(F_p, 0.0); fill!(K_p, 0.0)
        asm_k = start_assemble(K_int, r_int)
        asm_p = start_assemble(K_p)
        for cell in CellIterator(dh)
            fill!(ke, 0.0); fill!(re, 0.0)
            reinit!(scv, cell)
            sd  = shelldofs(cell)
            u_e = u[sd]
            membrane_residuals_RM!(re, scv, u_e, mat)
            bending_residuals_RM!(re, scv, u_e, mat)
            membrane_tangent_RM!(ke, scv, u_e, mat)
            bending_tangent_RM!(ke, scv, u_e, mat)
            assemble!(asm_k, sd, ke, re)
            fill!(ke, 0.0); fill!(re, 0.0)
            assemble_pressure!(re, scv, u_e, 1.0)
            assemble_pressure_tangent!(ke, scv, u_e, 1.0)
            assemble!(asm_p, sd, ke)
            @views F_p[sd] .+= re
        end

        R   = r_int .- p_max .* F_p
        apply_zero!(R, ch)
        norm(@views R[free]) < tol && return u, dh, scv, iter

        K_eff = K_int .- p_max .* K_p
        rhs   = .-R
        apply_zero!(K_eff, rhs, ch)
        u .+= K_eff \ rhs
        apply!(u, ch)
    end
    @warn "static_solve: no convergence in $max_iter iterations"
    return u, dh, scv, max_iter
end

mat = LinearElastic(1.0e6, 0.3, 0.009)
p   = 500.0

@printf("%-20s  %6s  %10s  %6s\n", "mesh", "ndofs", "w_center", "iters")

for n in [8, 16, 32]
    grid = make_pillow_gmsh_grid(; L=1.0, n)
    u, dh, scv, iters = static_solve(grid, mat, p)
    w_center = maximum(abs, u)
    @printf("%-20s  %6d  %10.4e  %6d\n", "gmsh n=$n", ndofs(dh), w_center, iters)
    VTKGridFile("pillow_gmsh_n$(n)", dh) do vtk
        write_solution(vtk, dh, u)
    end
end

grid_struct = begin
    g = make_quarter_pillow_grid(32; L=1.0, primitive=QuadraticQuadrilateral)
    addfacetset!(g, "edge",  x -> isapprox(x[1], 0.5, atol=1e-10) || isapprox(x[2], 0.5, atol=1e-10))
    addfacetset!(g, "sym_x", x -> isapprox(x[1], 0.0, atol=1e-10))
    addfacetset!(g, "sym_y", x -> isapprox(x[2], 0.0, atol=1e-10))
    addnodeset!(g, "center", x -> norm(x) < 1e-10)
    g
end
u, dh, _, iters = static_solve(grid_struct, mat, p)
@printf("%-20s  %6d  %10.4e  %6d\n", "structured 32×32", ndofs(dh), maximum(abs, u), iters)
VTKGridFile("pillow_structured_32x32", dh) do vtk
    write_solution(vtk, dh, u)
end
