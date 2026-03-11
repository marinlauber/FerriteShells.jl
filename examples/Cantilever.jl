using FerriteShells

function make_cantilever(dims; L=10.0, W=1.0)
    corners = [Vec{2}((0.0, 0.0)), Vec{2}((L, 0.0)), Vec{2}((L, W)), Vec{2}((0.0, W))]
    grid = shell_grid(generate_grid(QuadraticQuadrilateral, dims, corners))
    Δx = L / dims[1]
    addfacetset!(grid, "clamped",  x -> x[1] ≈ 0)
    addfacetset!(grid, "traction", x -> x[1] ≈ L)
    addnodeset!(grid, "all",      x -> true)
    addfacetset!(grid, "dθ", x -> x[1] ≤ Δx + 1e-10)  # x=0, Δx/2, Δx
    return grid
end

function assemble_global_shell!(K, g, u, dh, scv, mat)
    n_e = ndofs_per_cell(dh)
    ke  = zeros(n_e, n_e)
    re  = zeros(n_e)
    asm = start_assemble(K, g)
    for cell in CellIterator(dh)
        fill!(ke, 0.0); fill!(re, 0.0)
        reinit!(scv, cell)
        u_e = u[celldofs(cell)]
        membrane_tangent_KL!(ke, scv, u_e, mat)
        bending_tangent_KL!(ke, scv, u_e, mat)
        membrane_residuals_KL!(re, scv, u_e, mat)
        bending_residuals_KL!(re, scv, u_e, mat)
        assemble!(asm, celldofs(cell), ke, re)
    end
end

# make a mesh
grid = make_cantilever((16,4))

# material model and thickness,
# take from M. Neunteufel, J. Schöberl / Computers and Structures 225 (2019) 106109
mat = LinearElastic(1.2e6, 0.0, 0.1)

# interpolation order
ip = Lagrange{RefQuadrilateral, 2}()
qr = QuadratureRule{RefQuadrilateral}(3)

# cell (shell) values
scv = ShellCellValues(qr, ip, ip)
fqr = FacetQuadratureRule{RefQuadrilateral}(3)

# degrees of freedom for displacements (pure membrane test)
dh = DofHandler(grid)
add!(dh, :u, ip^3)
close!(dh)

# boundary conditions
dbc = ConstraintHandler(dh)
add!(dbc, Dirichlet(:u, getfacetset(dh.grid, "clamped"), x -> zero(x), [1,2,3]))
add!(dbc, Dirichlet(:u, getfacetset(dh.grid,  "dθ"), x -> 0.0, [3]))
close!(dbc)

# Pre-allocation of vectors for the solution and Newton increments
_ndofs = ndofs(dh)
un = zeros(_ndofs) # previous solution vector
u = zeros(_ndofs)
Δu = zeros(_ndofs)

# Create sparse matrix and residual vector
K = allocate_matrix(dh)
g = zeros(_ndofs)

using WriteVTK
pvd = paraview_collection("cantilever")
VTKGridFile("cantilever-0", dh) do vtk
    write_solution(vtk, dh, u); pvd[0] = vtk
end

# traction is contant along load step, assemble once
traction = Vec{3}((0., 0., 0.01))
f_ext = zeros(ndofs(dh))
assemble_traction!(f_ext, dh, getfacetset(grid, "traction"), ip, fqr, traction)

# solve
let newton_itr = 0; @time while true
    newton_itr += 1
    # Construct the current guess
    u .= un .+ Δu
    # Compute residual and tangent for current guess
    assemble_global_shell!(K, g, u, dh, scv, mat)
    g .-= f_ext # apply external force
    # Apply boundary conditions
    apply_zero!(K, g, dbc)
    # Compute the residual norm and compare with tolerance
    norm(g[dbc.free_dofs]) < 1e-6 && break
    newton_itr > 30 && break
    # Compute increment
    Δu .-= K \ g
    # make sure BC are zero
    apply_zero!(Δu, dbc)
    # save
    VTKGridFile("cantilever-$newton_itr", dh) do vtk
        write_solution(vtk, dh, u); pvd[newton_itr] = vtk
    end
end; println("Converged in $newton_itr iterations to $(norm(g[dbc.free_dofs]))")
end
close(pvd);