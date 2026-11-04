include(joinpath(@__DIR__, "rm_d3.jl")); using Printf
const R, L, Φ = 25.0, 50.0, 40π/180
function roof(C, N)
    g2 = generate_grid(C, (N, N), Vec{2}((-Φ, 0.0)), Vec{2}((Φ, L)))
    g = Grid(getcells(g2), [Node(Vec{3}((x[2], R*cos(x[1]), R*sin(x[1])))) for x in get_node_coordinate.(getnodes(g2))])
    addnodeset!(g, "diaphragm", x -> x[1] ≈ 0.0 || x[1] ≈ L)
    addnodeset!(g, "ref", x -> abs(x[1]-L/2)<1e-8 && abs(x[2]-R*cos(Φ))<1e-8 && abs(x[3]-R*sin(Φ))<1e-8)
    g
end
function solve_roof(C, ip, qr, N)
    grid = roof(C, N)
    s = RMShell(ip, ip, qr; E = 4.32e8, ν = 0.0, t = 0.25)
    dh = DofHandler(grid); add!(dh, :u, ip^3); add!(dh, :ω, ip^3); close!(dh)
    Dn = nodal_normals(grid, ip)
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getnodeset(grid, "diaphragm"), x -> zeros(2), [2, 3]))
    add!(ch, Dirichlet(:u, getnodeset(grid, "ref"), x -> 0.0, [1]))
    drilling_constraints!(ch, dh, Dn); close!(ch)
    K, r, f = allocate_matrix(dh, ch), zeros(ndofs(dh)), zeros(ndofs(dh))
    u = zeros(ndofs(dh))
    t_asm = @elapsed assemble_global!(K, r, dh, s, Dn, u)
    q = Vec{3}((0.0, -90.0, 0.0)); cv = s.cv
    for cell in CellIterator(dh)
        reinit!(cv, getcoordinates(cell)); cd = celldofs(cell)
        for qp in 1:getnquadpoints(cv), i in 1:getnbasefunctions(cv.u)
            f[cd[i]] += shape_value(cv.u, qp, i) ⋅ q * getdetJdV(cv, qp)
        end
    end
    rhs = f - r; apply!(K, rhs, ch); u = K \ rhs; apply!(u, ch)
    rn = only(getnodeset(grid, "ref"))
    for cell in CellIterator(dh), (a, nid) in enumerate(getnodes(cell))
        nid == rn && return -u[celldofs(cell)[3a-1]] / 0.3024, ndofs(dh), t_asm
    end
end
for (name, C, ip, qr) in (("Q4 (D3)", Quadrilateral, Lagrange{RefQuadrilateral,1}(), QuadratureRule{RefQuadrilateral}(2)),
                          ("Q9 (D3)", QuadraticQuadrilateral, Lagrange{RefQuadrilateral,2}(), QuadratureRule{RefQuadrilateral}(3)))
    solve_roof(C, ip, qr, 4)
    res = [solve_roof(C, ip, qr, n) for n in (4, 8, 16, 32)]
    @printf("%-10s %s   n=32 ndofs=%d asm=%.3fs\n", name, join([@sprintf("%.10f", r[1]) for r in res], " "), res[end][2], res[end][3])
end
