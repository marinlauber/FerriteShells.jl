# using Ferrite, FerriteGmsh, SparseArrays
# using Downloads: download
# logo_mesh = "logo.geo"
# asset_url = "https://raw.githubusercontent.com/Ferrite-FEM/Ferrite.jl/gh-pages/assets/"
# isfile(logo_mesh) || download(string(asset_url, logo_mesh), logo_mesh)
# grid = togrid(logo_mesh);
# addfacetset!(grid, "top", x -> x[2] ≈ 1.0) # facets for which x[2] ≈ 1.0 for all nodes
# addfacetset!(grid, "left", x -> abs(x[1]) < 1.0e-6)
# addfacetset!(grid, "bottom", x -> abs(x[2]) < 1.0e-6);
# dim = 2
# order = 1 # linear interpolation
# ip = Lagrange{RefTriangle, order}()^dim; # vector valued interpolation
# qr = QuadratureRule{RefTriangle}(1) # 1 quadrature point
# facet_qr = FacetQuadratureRule{RefTriangle}(1);
# cellvalues = CellValues(qr, ip)
# facetvalues = FacetValues(facet_qr, ip);
# dh = DofHandler(grid)
# add!(dh, :u, ip)
# close!(dh);
# ch = ConstraintHandler(dh)
# add!(ch, Dirichlet(:u, getfacetset(grid, "bottom"), (x, t) -> 0.0, 2))
# add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> 0.0, 1))
# close!(ch);
# traction(x) = Vec(0.0, 20.0e3 * x[1]);
# function assemble_external_forces!(f_ext, dh, facetset, facetvalues, prescribed_traction)
#     # Create a temporary array for the facet's local contributions to the external force vector
#     fe_ext = zeros(getnbasefunctions(facetvalues))
#     for facet in FacetIterator(dh, facetset)
#         # Update the facetvalues to the correct facet number
#         reinit!(facetvalues, facet)
#         # Reset the temporary array for the next facet
#         fill!(fe_ext, 0.0)
#         # Access the cell's coordinates
#         cell_coordinates = getcoordinates(facet)
#         for qp in 1:getnquadpoints(facetvalues)
#             # Calculate the global coordinate of the quadrature point.
#             x = spatial_coordinate(facetvalues, qp, cell_coordinates)
#             tₚ = prescribed_traction(x)
#             # Get the integration weight for the current quadrature point.
#             dΓ = getdetJdV(facetvalues, qp)
#             for i in 1:getnbasefunctions(facetvalues)
#                 Nᵢ = shape_value(facetvalues, qp, i)
#                 fe_ext[i] += tₚ ⋅ Nᵢ * dΓ
#             end
#         end
#         # Add the local contributions to the correct indices in the global external force vector
#         assemble!(f_ext, celldofs(facet), fe_ext)
#     end
#     return f_ext
# end
# Emod = 200.0e3 # Young's modulus [MPa]
# ν = 0.3        # Poisson's ratio [-]
# Gmod = Emod / (2(1 + ν))  # Shear modulus
# Kmod = Emod / (3(1 - 2ν)) # Bulk modulus
# C = gradient(ϵ -> 2 * Gmod * dev(ϵ) + 3 * Kmod * vol(ϵ), zero(SymmetricTensor{2, 2}));
# function assemble_cell!(ke, cellvalues, C)
#     for q_point in 1:getnquadpoints(cellvalues)
#         # Get the integration weight for the quadrature point
#         dΩ = getdetJdV(cellvalues, q_point)
#         for i in 1:getnbasefunctions(cellvalues)
#             # Gradient of the test function
#             ∇Nᵢ = shape_gradient(cellvalues, q_point, i)
#             for j in 1:getnbasefunctions(cellvalues)
#                 # Symmetric gradient of the trial function
#                 ∇ˢʸᵐNⱼ = shape_symmetric_gradient(cellvalues, q_point, j)
#                 ke[i, j] += (∇Nᵢ ⊡ C ⊡ ∇ˢʸᵐNⱼ) * dΩ
#             end
#         end
#     end
#     return ke
# end
# function assemble_global!(K, dh, cellvalues, C)
#     # Allocate the element stiffness matrix
#     n_basefuncs = getnbasefunctions(cellvalues)
#     ke = zeros(n_basefuncs, n_basefuncs)
#     # Create an assembler
#     assembler = start_assemble(K)
#     # Loop over all cells
#     for cell in CellIterator(dh)
#         # Update the shape function gradients based on the cell coordinates
#         reinit!(cellvalues, cell)
#         # Reset the element stiffness matrix
#         fill!(ke, 0.0)
#         # Compute element contribution
#         assemble_cell!(ke, cellvalues, C)
#         # Assemble ke into K
#         assemble!(assembler, celldofs(cell), ke)
#     end
#     return K
# end
# K = allocate_matrix(dh)
# assemble_global!(K, dh, cellvalues, C);
# f_ext = zeros(ndofs(dh))
# assemble_external_forces!(f_ext, dh, getfacetset(grid, "top"), facetvalues, traction);
# apply!(K, f_ext, ch)
# u = K \ f_ext;
# function calculate_stresses(grid, dh, cv, u, C)
#     qp_stresses = [
#         [zero(SymmetricTensor{2, 2}) for _ in 1:getnquadpoints(cv)]
#             for _ in 1:getncells(grid)
#     ]
#     avg_cell_stresses = tuple((zeros(getncells(grid)) for _ in 1:3)...)
#     for cell in CellIterator(dh)
#         reinit!(cv, cell)
#         cell_stresses = qp_stresses[cellid(cell)]
#         for q_point in 1:getnquadpoints(cv)
#             ε = function_symmetric_gradient(cv, q_point, u, celldofs(cell))
#             cell_stresses[q_point] = C ⊡ ε
#         end
#         σ_avg = sum(cell_stresses) / getnquadpoints(cv)
#         avg_cell_stresses[1][cellid(cell)] = σ_avg[1, 1]
#         avg_cell_stresses[2][cellid(cell)] = σ_avg[2, 2]
#         avg_cell_stresses[3][cellid(cell)] = σ_avg[1, 2]
#     end
#     return qp_stresses, avg_cell_stresses
# end
# qp_stresses, avg_cell_stresses = calculate_stresses(grid, dh, cellvalues, u, C);
# proj = L2Projector(Lagrange{RefTriangle, 1}(), grid)
# stress_field = project(proj, qp_stresses, qr);
# VTKGridFile("linear_elasticity", dh) do vtk
#     write_solution(vtk, dh, u)
#     for (i, key) in enumerate(("11", "22", "12"))
#         write_cell_data(vtk, avg_cell_stresses[i], "sigma_" * key)
#     end
#     write_projection(vtk, proj, stress_field, "stress field")
#     Ferrite.write_cellset(vtk, grid)
# end

# shell version
#
# Key API differences vs. the plane-stress solid case above:
#   using Ferrite, FerriteGmsh          -> using FerriteShells, FerriteGmsh
#   grid = togrid(logo_mesh)            -> grid = shell_grid(togrid(logo_mesh))    # embed the 2D mesh in 3D (flat, z=0 plane)
#   ip = Lagrange{RefTriangle,1}()^dim  -> ip = Lagrange{RefTriangle,1}()          # scalar ip; :u (3 comp.) and :θ (2 comp.) are built from it below
#   cellvalues = CellValues(qr, ip)     -> scv = ShellCellValues(qr, ip, ip)       # geometry + shape interpolation, works directly on Vec{3} nodes
#   facetvalues = FacetValues(...)      -> (not needed; assemble_traction! integrates directly on the embedded facet coordinates)
#   dh has one :u field (dim components) -> dh has :u (3 translations) + :θ (2 director rotations) per node
#   C = gradient(...)                   -> mat = LinearElastic(Emod, ν, thickness) # precomputed plane-stress elasticity tensor
#   assemble_cell!/assemble_global!     -> membrane_tangent_RM!/bending_tangent_RM!/membrane_residuals_RM!/bending_residuals_RM! via shelldofs(cell)
#   assemble_external_forces!           -> assemble_traction!(f_ext, dh, facetset, ip, fqr, traction) # traction still x::Vec -> Vec, now 3-component
#   calculate_stresses (σ = C ⊡ ε)      -> shell_strains(scv, qp, u_e) gives the membrane/bending/shear strain measures (E, κ, γ) instead
#
# The plate only carries in-plane loading (roller BCs on "left"/"bottom", in-plane traction on
# "top"), so nothing excites the out-of-plane response: u_z and the two director rotations θ are
# left completely free by those BCs (a uniform out-of-plane translation/tilt costs zero energy),
# which makes K singular. A single extra pin of (u_z, θ) at one node ("corner") removes exactly
# those 3 spurious rigid-body modes without otherwise touching the membrane solution.
using FerriteShells, FerriteGmsh, SparseArrays
using Downloads: download
logo_mesh = "logo.geo"
asset_url = "https://raw.githubusercontent.com/Ferrite-FEM/Ferrite.jl/gh-pages/assets/"
isfile(logo_mesh) || download(string(asset_url, logo_mesh), logo_mesh)
grid = shell_grid(togrid(logo_mesh)); # embed the 2D mesh into 3D (flat, z=0 plane)
addfacetset!(grid, "top", x -> x[2] ≈ 1.0) # facets for which x[2] ≈ 1.0 for all nodes
addfacetset!(grid, "left", x -> abs(x[1]) < 1.0e-6)
addfacetset!(grid, "bottom", x -> abs(x[2]) < 1.0e-6)
addnodeset!(grid, "corner", x -> abs(x[1]) < 1.0e-6 && abs(x[2]) < 1.0e-6); # the (0,0) node, used to pin the out-of-plane rigid modes
order = 1 # linear interpolation
ip = Lagrange{RefTriangle, order}()
qr = QuadratureRule{RefTriangle}(1) # 1 quadrature point
fqr = FacetQuadratureRule{RefTriangle}(1);
scv = ShellCellValues(qr, ip, ip)
dh = DofHandler(grid)
add!(dh, :u, ip^3) # translations
add!(dh, :θ, ip^2) # director rotations
close!(dh);
ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "bottom"), (x, t) -> 0.0, 2))
add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> 0.0, 1))
add!(ch, Dirichlet(:u, getnodeset(grid, "corner"), x -> 0.0, [3]))         # pin u_z ...
add!(ch, Dirichlet(:θ, getnodeset(grid, "corner"), x -> zeros(2), [1, 2])) # ... and θ at one node
close!(ch);
traction(x) = Vec{3}((0.0, 20.0e3 * x[1], 0.0));
Emod = 200.0e3 # Young's modulus [MPa]
ν = 0.3        # Poisson's ratio [-]
mat = LinearElastic(Emod, ν, 1.0) # thickness = 1.0, matching the implicit unit-thickness plane-stress assumption above
function assemble_global!(K, dh, scv, mat)
    # Allocate the element stiffness/residual arrays
    n = ndofs_per_cell(dh)
    ke = zeros(n, n)
    re = zeros(n)
    u0 = zeros(n) # linear problem: tangent/residual only need to be evaluated once, at u=0
    # Create an assembler
    assembler = start_assemble(K, zeros(ndofs(dh)))
    # Loop over all cells
    for cell in CellIterator(dh)
        # Reset the element stiffness matrix
        fill!(ke, 0.0); fill!(re, 0.0)
        # Update the reference geometry based on the cell coordinates
        reinit!(scv, cell)
        # Compute the membrane and bending (incl. transverse shear) contributions
        membrane_tangent_RM!(ke, scv, u0, mat)
        bending_tangent_RM!(ke, scv, u0, mat)
        membrane_residuals_RM!(re, scv, u0, mat)
        bending_residuals_RM!(re, scv, u0, mat)
        # Assemble ke into K, using the interleaved 5-DOF-per-node ordering
        assemble!(assembler, shelldofs(cell), ke, re)
    end
    return K
end
K = allocate_matrix(dh)
assemble_global!(K, dh, scv, mat);
f_ext = zeros(ndofs(dh))
assemble_traction!(f_ext, dh, getfacetset(grid, "top"), ip, fqr, traction);
apply!(K, f_ext, ch)
u = K \ f_ext;
# shell_strains returns E in covariant components on the element's own parametric basis
# (A₁, A₂), which is NOT the same direction from element to element on an unstructured
# mesh like this one (unlike the axis-aligned grids in the other examples). Averaging
# those raw components across elements during the L2 projection below would mix strain
# expressed in different, arbitrarily rotated frames. Since this shell is flat, T₁_elem
# and T₂_elem (computed once per element via reinit!) are the *same* global x̂, ŷ for
# every element regardless of its local numbering, so re-expressing E in that basis
# (via the metric) gives physically comparable, consistent Cartesian components first.
function to_cartesian(E, scv, qp)
    ginv   = inv(scv.A_metric[qp])
    A1, A2 = scv.A₁[qp], scv.A₂[qp]
    T1, T2 = scv.T₁_elem[1], scv.T₂_elem[1]
    b1 = ginv ⋅ Vec{2}((T1 ⋅ A1, T1 ⋅ A2))
    b2 = ginv ⋅ Vec{2}((T2 ⋅ A1, T2 ⋅ A2))
    return SymmetricTensor{2, 2}((b1 ⋅ (E ⋅ b1), b1 ⋅ (E ⋅ b2), b2 ⋅ (E ⋅ b2)))
end
function compute_strains(dh, scv, u)
    n_qp    = getnquadpoints(scv)
    n_cells = getncells(dh.grid)
    E_mem = [Vector{SymmetricTensor{2, 3, Float64, 6}}(undef, n_qp) for _ in 1:n_cells]
    for cell in CellIterator(dh)
        reinit!(scv, cell)
        u_e = u[shelldofs(cell)]
        id  = cellid(cell)
        for qp in 1:n_qp
            E, _, _ = shell_strains(scv, qp, u_e) # membrane, bending, and shear strains at this qp
            E_mem[id][qp] = embed23(to_cartesian(E, scv, qp))
        end
    end
    return E_mem
end
E_mem = compute_strains(dh, scv, u);
proj = L2Projector(ip, grid)
strain_field = project(proj, E_mem, qr);
VTKGridFile("linear_elasticity_shell", dh) do vtk
    write_solution(vtk, dh, u)
    write_projection(vtk, proj, strain_field, "E_membrane")
    Ferrite.write_cellset(vtk, grid)
end