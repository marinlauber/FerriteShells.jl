using FerriteShells, SparseArrays, OhMyThreads, TaskLocalValues

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

struct ScratchData{SCC, SCV, T, A}
    cell_cache::SCC
    scv::SCV
    ke::Matrix{T}
    fe::Vector{T}
    assembler::A
end

function ScratchData(dh::DofHandler, scv::ShellCellValues, K::SparseMatrixCSC,
                     f::Vector, ::Val{atomic} = Val(false)) where {atomic}
    cell_cache = CellCache(dh)
    n = ndofs_per_cell(dh)
    ke = zeros(eltype(K), n, n)
    fe = zeros(eltype(f), n)
    asm = start_assemble(K, f; fillzero=false, atomic=atomic)
    return ScratchData(cell_cache, copy(scv), ke, fe, asm)
end

function assemble_cell!(scratch::ScratchData, u, material, cellidx)
    (; cell_cache, scv, ke, fe, assembler) = scratch
    reinit!(cell_cache, cellidx)
    reinit!(scv, cell_cache)
    fill!(ke, 0.0); fill!(fe, 0.0)
    u_e = u[shelldofs(cell_cache)]
    membrane_tangent_RM!(ke, scv, u_e, material); bending_tangent_RM!(ke, scv, u_e, material)
    membrane_residuals_RM!(fe, scv, u_e, material); bending_residuals_RM!(fe, scv, u_e, material)
    assemble!(assembler, shelldofs(cell_cache), ke, fe)
end

function assemble_membrane_atomic!(K, r, dh, cellvalue_template::ShellCellValues, u, material, ntasks=Threads.nthreads())
    _ = start_assemble(K, r)
    scheduler = OhMyThreads.DynamicScheduler(; ntasks)
    OhMyThreads.@tasks for cellidx in 1:getncells(dh.grid)
        @set scheduler = scheduler
        @local scratch = ScratchData(dh, cellvalue_template, K, r, Val(true))
        assemble_cell!(scratch, u, material, cellidx)   # function barrier
    end
    return K, r
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
grid = create_cook_grid(8*32, 8*16; primitive=QuadraticQuadrilateral)

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
u = zeros(ndofs(dh))

# Sanity check: the multi-threaded assembly must reproduce the serial one bit-for-bit
# up to floating-point summation order. Assemble both into separate buffers and compare
# before trusting the parallel path — a data race would otherwise pass unnoticed.
Ks = allocate_matrix(dh); fs = zeros(ndofs(dh))
@time assemble_membrane!(Ks, fs, dh, scv, u, mat)          # serial reference
@time assemble_membrane_atomic!(Ke, f, dh, scv, u, mat)    # multi-threaded
@assert Ke ≈ Ks "threaded stiffness disagrees with serial assembly"
@assert f  ≈ fs "threaded residual disagrees with serial assembly"

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