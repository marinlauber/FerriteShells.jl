using FerriteShells,LinearAlgebra

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

# domain ω ∈ ]-1/2; 1/2[ and 3D grid
grid = shell_grid(generate_grid(QuadraticQuadrilateral, (40, 40), Vec(-0.5, -0.5), Vec(0.5, 0.5));
                  map=(n)->(100*n.x[1], 100*n.x[2], 100*(n.x[1]^2-n.x[2]^2)))

# add the Dirichlet Boundary on the faces of the model
addfacetset!(grid, "clamped",  x -> x[1] ≈  50.0)
addfacetset!(grid, "traction", x -> x[1] ≈ -50.0)

# interpolation order
ip = Lagrange{RefQuadrilateral, 2}() # Q9
qr = QuadratureRule{RefQuadrilateral}(3)

# cell (shell) values
scv = ShellCellValues(qr, ip, ip)
fqr = FacetQuadratureRule{RefQuadrilateral}(3)

# Linear elastic material
E = 200000.0 # MPa
ν = 0.30     # n-a
t = 2.0      # thickness in mm
mat = LinearElastic(E, ν, t)

# degrees of freedom
dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

# add the boundary condition to the dh
dbc = ConstraintHandler(dh)
add!(dbc, Dirichlet(:u, getfacetset(grid, "clamped"), x -> zero(x), [1,2,3]))
add!(dbc, Dirichlet(:θ, getfacetset(grid, "clamped"), x -> zeros(2), [1,2]))
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
pvd = paraview_collection("hyperbolic_paraboloid")
VTKGridFile("hyperbolic_paraboloid-0", dh) do vtk
    write_solution(vtk, dh, u); pvd[0] = vtk
end

# traction is constant along load step, assemble once
traction = Vec{3}((0., 0., -40.0))
f_ext = zeros(ndofs(dh))
assemble_traction!(f_ext, dh, getfacetset(grid, "traction"), ip, fqr, traction)

# solve with Newton-Raphson
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
#     VTKGridFile("hyperbolic_paraboloid-$newton_itr", dh) do vtk
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
    VTKGridFile("hyperbolic_paraboloid-$newton_itr", dh) do vtk
        write_solution(vtk, dh, u); pvd[newton_itr] = vtk
    end
end
end
close(pvd);