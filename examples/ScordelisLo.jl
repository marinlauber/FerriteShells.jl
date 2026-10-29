using FerriteShells

# helper for the mesh
function scordelis_lo_rm_grid(N)
    R_sl, L_sl, Φ_sl = 25.0, 50.0, 40π/180
    g = shell_grid(
        generate_grid(QuadraticQuadrilateral, (N, N),
                      Vec{2}((-Φ_sl, 0.0)), Vec{2}((Φ_sl, L_sl)));
        map = n -> (n.x[2], R_sl * cos(n.x[1]), R_sl * sin(n.x[1])))
    addnodeset!(g, "diaphragm", x -> x[1] ≈ 0.0 || x[1] ≈ L_sl)
    addnodeset!(g, "ref_point",
        x -> abs(x[1] - L_sl/2) < 1e-8 && abs(x[2] - R_sl*cos(Φ_sl)) < 1e-8 &&
             abs(x[3] - R_sl*sin(Φ_sl)) < 1e-8)
    return g
end

# interpolation scape and material model
ip  = Lagrange{RefQuadrilateral, 2}()
qr  = QuadratureRule{RefQuadrilateral}(3)
scv = ShellCellValues(qr, ip, ip)
mat = LinearElastic(4.32e8, 0.0, 0.25)

# make the mesh and degrees of freedom
grid = scordelis_lo_rm_grid(32)
dh   = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

# allocation
n_el   = ndofs_per_cell(dh)
n_base = getnbasefunctions(ip)
K  = allocate_matrix(dh)
f  = zeros(ndofs(dh))

# assembly once
asmb = start_assemble(K, zeros(ndofs(dh)))
ke = zeros(5n_base, 5n_base); re = zeros(5n_base); fe = zeros(5n_base)
q_sl = Vec{3}((0.0, -90.0, 0.0))
for cell in CellIterator(dh)
    fill!(ke, 0.0); fill!(re, 0.0); fill!(fe, 0.0)
    reinit!(scv, cell)
    u0 = zeros(5n_base)
    membrane_tangent_RM!(ke, scv, u0, mat)
    bending_tangent_RM!(ke, scv, u0, mat)
    sd = shelldofs(cell)
    assemble!(asmb, sd, ke, re)
    for qp in 1:getnquadpoints(scv)
        ξ  = scv.qr.points[qp]; dΩ = scv.detJdV[qp]
        for I in 1:n_base
            NI = Ferrite.reference_shape_value(ip, ξ, I)
            @views fe[5I-4:5I-2] .+= NI * q_sl * dΩ
        end
    end
    @views f[sd] .+= fe
end

# boundary conditions
dbc = ConstraintHandler(dh)
add!(dbc, Dirichlet(:u, getnodeset(grid, "diaphragm"), x -> zeros(2), [2, 3]))
# The diaphragms leave the axial u_x free, so K keeps a rigid-body translation along x
# (cond ~1e17). u_x = 0 holds exactly at mid-span by symmetry; "ref_point" is a mid-span node.
add!(dbc, Dirichlet(:u, getnodeset(grid, "ref_point"), x -> 0.0, [1]))
close!(dbc); Ferrite.update!(dbc, 0.0); apply!(K, f, dbc)

# solve
@time u_sol = K \ f

# write to vtk
VTKGridFile("scordelis_Lo_roof", dh) do vtk
    write_solution(vtk, dh, u_sol)
end

# get the solution
ref_nodes = collect(getnodeset(grid, "ref_point"))
sol = []
@assert length(ref_nodes) == 1
for cell in CellIterator(dh)
    for (I, gid) in enumerate(getnodes(cell))
        if gid == ref_nodes[1]
            cd = celldofs(cell)
            push!(sol, u_sol[cd[3I-1]])  # y-component of :u
        end
    end
end
# get solution
w = first(sol)
println("Scordelis-Lo: u_y at free-edge midpoint = $(round(w; digits=5)) (reference: -0.3024)")
