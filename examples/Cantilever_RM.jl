using FerriteShells

function make_cantilever(dims; L=10.0, W=1.0)
    corners = [Vec{2}((0.0, 0.0)), Vec{2}((L, 0.0)), Vec{2}((L, W)), Vec{2}((0.0, W))]
    grid = shell_grid(generate_grid(QuadraticQuadrilateral, dims, corners))
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
        u_e = u[shelldofs(cell)]
        FerriteShells.membrane_tangent_RM_explicit!(ke, scv, u_e, mat)
        FerriteShells.bending_tangent_RM_explicit!(ke, scv, u_e, mat)
        FerriteShells.membrane_residuals_RM_explicit!(re, scv, u_e, mat)
        FerriteShells.bending_residuals_RM_explicit!(re, scv, u_e, mat)
        assemble!(asm, shelldofs(cell), ke, re)
    end
end

# make a mesh
grid = make_cantilever((32,4))

# boundaries
addfacetset!(grid, "clamped",  x -> x[1] ≈ 0)
addfacetset!(grid, "traction", x -> x[1] ≈ 10)

# material model and thickness,
# take from M. Neunteufel, J. Schöberl / Computers and Structures 225 (2019) 106109
mat = LinearElastic(1.2e6, 0.0, 0.1)

# interpolation order
ip = Lagrange{RefQuadrilateral, 1}() # Q9
qr = QuadratureRule{RefQuadrilateral}(3)

# cell (shell) values
scv = ShellCellValues(qr, ip, ip)
fqr = FacetQuadratureRule{RefQuadrilateral}(3)

# degrees of freedom for displacements (pure membrane test)
dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

# boundary conditions
dbc = ConstraintHandler(dh)
add!(dbc, Dirichlet(:u, getfacetset(dh.grid, "clamped"), x -> zero(x), [1,2,3]))
add!(dbc, Dirichlet(:θ, getfacetset(dh.grid, "clamped"), x -> zeros(2), [1,2]))
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

# traction is constant along load step, assemble once
traction = Vec{3}((0., 0., 4.0))
f_ext = zeros(ndofs(dh))
assemble_traction!(f_ext, dh, getfacetset(grid, "traction"), ip, fqr, traction)

# brute force Newton solve for initial load step λ=1.0, save intermediate steps for visualization
# let newton_itr = 0; @time while true
#     newton_itr += 1
#     # Construct the current guess
#     u .= un .+ Δu
#     # Compute residual and tangent for current guess
#     assemble_global_shell!(K, g, u, dh, scv, mat)
#     g .-= f_ext # apply external force
#     # Apply boundary conditions
#     apply_zero!(K, g, dbc)
#     # Compute the residual norm and compare with tolerance
#     norm(g[dbc.free_dofs]) < 1e-6 && break
#     newton_itr > 30 && break
#     # Compute increment
#     Δu .-= K \ g
#     # make sure BC are zero
#     apply_zero!(Δu, dbc)
#     # save
#     VTKGridFile("cantilever-$newton_itr", dh) do vtk
#         write_solution(vtk, dh, u); pvd[newton_itr] = vtk
#     end
# end; println("Converged in $newton_itr iterations to $(norm(g[dbc.free_dofs]))")
# end
# close(pvd);

# load controlled Newton-Raphson
let λᵢ=0; @time for λ in 0.2:0.2:1.0
    # Newton solve for current load step
    λᵢ += 1; newton_itr = 0
    while true
        newton_itr += 1
        # Construct the current guess
        u .= un .+ Δu
        # Compute residual and tangent for current guess
        assemble_global_shell!(K, g, u, dh, scv, mat)
        g .-= λ .* f_ext # apply external force
        # Apply boundary conditions
        apply_zero!(K, g, dbc)
        # Compute the residual norm and compare with tolerance
        norm(g[dbc.free_dofs]) < 1e-6 && break
        newton_itr > 30 && break
        # Compute increment
        Δu .-= K \ g
        # make sure BC are zero
        apply_zero!(Δu, dbc)
    end
    println("Load step λ=$(round(λ; digits=2)) converged in $newton_itr iterations to $(norm(g[dbc.free_dofs]))")
    # save
    VTKGridFile("cantilever-$λᵢ", dh) do vtk
        write_solution(vtk, dh, u); pvd[λᵢ] = vtk
    end
end;
end
close(pvd);


# # arc-length controlled Newton-Raphson
# let λᵢ=0; λ=0.1; @time while λ < 1.0
#     # Newton solve for current load step
#     λᵢ += 1; newton_itr = 0
#     while true
#         newton_itr += 1
#         # Construct the current guess
#         u .= un .+ Δu
#         # Compute residual and tangent for current guess
#         assemble_global_shell!(K, g, u, dh, scv, mat)
#         # compute the effective tangent and residual for the current guess
#         g .-= λ .* f_ext # apply external force
#         # Apply boundary conditions
#         apply_zero!(K, g, dbc)
#         # Compute the residual norm and compare with tolerance
#         norm(g[dbc.free_dofs]) < 1e-6 && break
#         newton_itr > 30 && break
#         # Compute increment
#         Δu .-= K \ g
#         # make sure BC are zero
#         apply_zero!(Δu, dbc)
#     end
#     println("Load step λ=$(round(λ; digits=2)) converged in $newton_itr iterations to $(norm(g[dbc.free_dofs]))")
#     # save
#     VTKGridFile("cantilever-$λᵢ", dh) do vtk
#         write_solution(vtk, dh, u); pvd[λᵢ] = vtk
#     end
# end;
# end
# close(pvd);