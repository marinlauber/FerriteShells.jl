using FerriteShells

function create_cook_grid(nx, ny; primitive=Quadrilateral)
    corners = [Vec{2}(( 0.0,  0.0)), Vec{2}((48.0, 44.0)),
               Vec{2}((48.0, 60.0)), Vec{2}(( 0.0, 44.0))]
    return generate_grid(primitive, (nx, ny), corners) |> shell_grid # embed in into a 3D space
end

function assemble_membrane!(K, r, dh, scv, u, mat)
    n = ndofs_per_cell(dh)
    ke = zeros(n, n)
    re  = zeros(n)
    assembler = start_assemble(K, r)
    for cell in CellIterator(dh)
        fill!(ke, 0.0); fill!(re, 0.0)
        reinit!(scv, cell) # prepares reference geometry
        u_e = u[shelldofs(cell)]
        membrane_tangent_RM!(ke, scv, u_e, mat)
        bending_tangent_RM!(ke, scv, u_e, mat)
        membrane_residuals_RM!(re, scv, u_e, mat)
        bending_residuals_RM!(re, scv, u_e, mat)
        assemble!(assembler, shelldofs(cell), ke, re)
    end
end

# helper to compute the membrane, bending, and shear strains at quadrature points for postprocessing
function compute_strains(dh, scv, u)
    n_qp    = getnquadpoints(scv)
    n_cells = getncells(dh.grid)
    E_mem     = [Vector{SymmetricTensor{2,3,Float64,6}}(undef, n_qp) for _ in 1:n_cells]
    kappa     = [Vector{SymmetricTensor{2,3,Float64,6}}(undef, n_qp) for _ in 1:n_cells]
    gamma     = [Vector{Vec{3,Float64}}(undef, n_qp) for _ in 1:n_cells]
    for cell in CellIterator(dh)
        reinit!(scv, cell)
        u_e = u[shelldofs(cell)]
        id  = cellid(cell)
        @inbounds for qp in 1:n_qp
            E, κ, γ = shell_strains(scv, qp, u_e)
            E_mem[id][qp]  = embed23(E)
            kappa[id][qp]  = embed23(κ)
            gamma[id][qp]  = Vec{3}((γ[1], γ[2], 0.0))
        end
    end
    E_mem, kappa, gamma
end

# number of cells
grid = create_cook_grid(32, 16; primitive=QuadraticQuadrilateral)

# facesets for boundary conditions
addfacetset!(grid,  "clamped", x -> norm(x[1]) ≈ 0.0)
addfacetset!(grid, "traction", x -> norm(x[1]) ≈ 48.0)

# interpolation order
# ip = Lagrange{RefQuadrilateral, 1}() # Q4
ip = Lagrange{RefQuadrilateral, 2}() # Q9
# ip = Lagrange{RefTriangle, 2}() # S3
qr = QuadratureRule{RefQuadrilateral}(3)

# cell (shell) values
scv = ShellCellValues(qr, ip, ip)
fqr = FacetQuadratureRule{RefQuadrilateral}(3)

# degrees of freedom for displacements and rotations
dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

# linear material model
mat = LinearElastic(1.0, 1/3, 1.0)

# hyperelastic material
μ = 1.0/3.0
mat = Hyperelastic(C->μ/2*(tr(C)-3), 1.0)

using UniversalMaterialModel
k₁  = 5.0
k₂  = 20.0
f₁  = Vec(1.0, 0.0, 0.0)
f₂  = Vec(0.0, 1.0, 0.0)
terms = [(1.0,1.0,1.0,1.0,1.0,1.0,μ/2.0),
         (4.0,2.0,2.0,2.0,1.0,k₂,k₁/2k₂),
         (8.0,2.0,2.0,2.0,1.0,k₂,k₁/2k₂)]
Holz = UniversalMaterialModel.build_material(terms)
mat = Hyperelastic(C->UniversalMaterialModel.Ψ(C, Holz; fibers=(f₁,f₂)), 1.0)

# boundary conditions
dbc = ConstraintHandler(dh)
add!(dbc, Dirichlet(:u, getfacetset(dh.grid, "clamped"), x -> zero(x), [1,2,3]))
add!(dbc, Dirichlet(:θ, getfacetset(dh.grid, "clamped"), x -> [0.0,0.0], [1,2]))
close!(dbc)

# projection operator
proj = L2Projector(ip, grid)

# stiffness matrix and residuals vector construction and assembly
Ke = allocate_matrix(dh)
f = zeros(ndofs(dh))
assemble_membrane!(Ke, f, dh, scv, zeros(ndofs(dh)), mat)

# traction force assembly, force of 1N on the face, split into 16 units (length of face)
assemble_traction!(f, dh, getfacetset(grid, "traction"), ip, fqr, (0.0, 1/16, 0.0))

# apply BCs and solve (\) figures out the best linear solver to use
apply!(Ke, f, dbc)
@time ue = Ke \ f

# extract solution at point
ph = PointEvalHandler(grid, [Vec{3}((48.0, 52.0, 0.0))])
u_eval = first(evaluate_at_points(ph, dh, ue, :u))
@show u_eval

# write to vtk
VTKGridFile("cooks_membrane", dh) do vtk
    write_solution(vtk, dh, ue)
    E_mem, κ, γ = compute_strains(dh, scv, ue)
    write_projection(vtk, proj, project(proj, E_mem, qr), "E_membrane")
    write_projection(vtk, proj, project(proj, κ,     qr), "kappa_bending")
    write_projection(vtk, proj, project(proj, γ,     qr), "gamma_shear")
end