using FerriteShells,LinearAlgebra,Printf,WriteVTK

# model parameters
L  = 1.0; q_z = 80.0

# material model, grid and boundary set
mat  = LinearElastic(2e11, 0.3, 0.01)
grid = shell_grid(generate_grid(QuadraticQuadrilateral, (200, 200),
                                Vec(-L/2, -L/2), Vec(L/2, L/2));
                  map = n -> (n.x[1], n.x[2], n.x[1]^2 - n.x[2]^2))
addfacetset!(grid, "clamped", x -> x[2] ≈ -L/2)

# interpolation and shell with shear treatment
ip  = Lagrange{RefQuadrilateral, 2}()
qr  = QuadratureRule{RefQuadrilateral}(3)
nf  = NodeFrames(grid, ip)
scv = ShellCellValues(qr, ip, ip; mitc=MITC9)

# degrees of freedom
dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

# boundary conditions
dbc = ConstraintHandler(dh)
add!(dbc, Dirichlet(:u, getfacetset(grid, "clamped"), x -> zero(x), [1,2,3]))
add!(dbc, Dirichlet(:θ, getfacetset(grid, "clamped"), x -> zeros(2), [1,2]))
close!(dbc)

# matrices and vectors
n_el   = ndofs_per_cell(dh)
n_base = getnbasefunctions(scv.ip_shape)
ke = zeros(n_el, n_el)
fe = zeros(n_el)
K     = allocate_matrix(dh)
f_ext = zeros(ndofs(dh))

# assembly
asm = start_assemble(K, f_ext)
for cell in CellIterator(dh)
    fill!(ke, 0.0); fill!(fe, 0.0)
    reinit!(scv, cell, nf)
    u_e = zeros(n_el)
    membrane_tangent_RM!(ke, scv, u_e, mat)
    bending_tangent_RM!(ke, scv, u_e, mat)
    for qp in 1:getnquadpoints(scv)
        dΩ = scv.detJdV[qp]; ξ = scv.qr.points[qp]
        for I in 1:n_base
            NI = Ferrite.reference_shape_value(scv.ip_shape, ξ, I)
            fe[5(I-1)+3] -= NI * q_z * dΩ   # u_z in shelldofs ordering
        end
    end
    assemble!(asm, shelldofs(cell), ke, fe)
end

# solution
f_ref = copy(f_ext)   # save before BCs overwrite f_ext
apply!(K, f_ext, dbc)
u = K \ f_ext

VTKGridFile("hyperbolic_paraboloid", dh) do vtk
    write_solution(vtk, dh, u)
end