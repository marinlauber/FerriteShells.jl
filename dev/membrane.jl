using FerriteShells

function plate_grid(nx, ny)
    corners = [Vec{2}((0.0,  0.0)), Vec{2}((1.0, 0.0)),
               Vec{2}((1.0, 1.0)), Vec{2}((0.0,  1.0))]
    return generate_grid(Quadrilateral, (nx, ny), corners) |> shell_grid
end


# function membrane_residuals!(re, scv, x, u_e, mat)
#     for qp in 1:getnquadpoints(scv)
#         # current kinematics
#         a₁,a₂,A_metric,a_metric = kinematics(scv, qp, x, u_e)
#         E = 0.5 * (a_metric - A_metric)
#         N = mat.H ⊡ E # ⊡ is the double contraction operator for 4th order tensors H_αβγδ E_γδ
#         dΩ = scv.detJdV[qp]
#         for I in 1:length(scv.N)
#             ∂NI1,∂NI2 = scv.∇N[qp, I]
#             # assemble residuals for this node I
#             v = ∂NI1 * (N[1,1]*a₁ + N[1,2]*a₂) +
#                 ∂NI2 * (N[2,1]*a₁ + N[2,2]*a₂)
#             # assemble into residual
#             re[3I-2:3I] .+= v * dΩ # checked, should be correct
#         end
#     end
#     return nothing
# end

# function membrane_tangent!(ke, scv, x, u_e, mat)
#     for qp in 1:getnquadpoints(scv)
#         # current kinematics
#         a₁,a₂,A_metric,a_metric = kinematics(scv, qp, x, u_e)
#         # Green-Lagrange strain is half the increment in metric tensor
#         E = 0.5 * (a_metric - A_metric)
#         N = mat.H ⊡ E # ⊡ is the double contraction operator for 4th order tensors H_αβγδ E_γδ
#         C = mat.C
#         @show a₁ a₂ A_metric a_metric E N C
#         dΩ = scv.detJdV[qp]
#         for I in 1:length(scv.N), J in 1:length(scv.N)
#             ∂NI1,∂NI2 = scv.∇N[qp, I]
#             ∂NJ1,∂NJ2 = scv.∇N[qp, J]
#             # geometric term
#             geo_scalar = ∂NI1*(N[1,1]*∂NJ1 + N[1,2]*∂NJ2) +
#                          ∂NI2*(N[2,1]*∂NJ1 + N[2,2]*∂NJ2)
#             Kgeo = geo_scalar * one(SymmetricTensor{2,3})
#             # big mess that looks ok on paper
#             Kmat = zero(SymmetricTensor{2,3})
#             H1 = SymmetricTensor{2,2}((∂NJ1, 0.5∂NJ2, 0))
#             H2 = SymmetricTensor{2,2}((0, 0.5∂NJ1, ∂NJ2))
#             D1 = C ⊡ H1
#             D2 = C ⊡ H2
#             for (α,∂NIα) in enumerate((∂NI1, ∂NI2)), (β,aβ) in enumerate((a₁, a₂))
#                 # vector result from scalar contraction
#                 v = D1[α,β]*a₁ + D2[α,β]*a₂
#                 Kmat += ∂NIα * (aβ ⊗ v)
#             end
#             # assemble
#             ke[3I-2:3I, 3J-2:3J] .+= (Kgeo + Kmat) * dΩ
#         end
#     end
#     return nothing
# end

# make the problem
n = 1
grid = plate_grid(n, n)
addfacetset!(grid, "clamped",  x -> x[1] ≈ 0.0)
addfacetset!(grid, "traction", x -> x[1] ≈ 1.0)

# interpolation
ip = Lagrange{RefQuadrilateral,1}()
# quadrature for membrane
qr = QuadratureRule{RefQuadrilateral}(1)

# membrane/bending and shear ShellCellValues
scv = ShellCellValues(qr, ip, ip)

# set degrees of freedom
dh = DofHandler(grid)
add!(dh, :u, ip^3)   # translational DOFs (u, v, w)
# add!(dh, :d, ip^3)   # director DOFs (d₁, d₂, d₃)
close!(dh)

# apply boundary conditions
dbc = ConstraintHandler(dh)
add!(dbc, Dirichlet(:u, getfacetset(dh.grid, "clamped"), x -> zero(x), [1, 2, 3]))
# add!(dbc, Dirichlet(:d, getfacetset(dh.grid, "clamped"), x -> [0.0, 0.0], [1, 2]))
close!(dbc)

# 40 Y. Ko et al. / Computers and Structures 192 (2017) 34–49 http://dx.doi.org/10.1016/j.compstruc.2017.07.003
E = 1.0        # stiffness (N)
t = 0.5        # thickness (dm)
ν = 1.0/3.0
mat = LinearElastic(E, ν, t)

# # select first cell
# cell = first(CellIterator(dh))
# reinit!(scv, cell) # prepares reference geometry

# # assemble and solve
# n = ndofs_per_cell(dh)
# ke = zeros(n, n)
# re  = zeros(n)
# T = Float64
# x = getcoordinates(cell)

# # zero displacement should be zero residual, check that first
# u = zeros(n) # passed: no displacements, should be zero residual for membrane part
# membrane_residuals!(re, scv, x, reinterpret(Vec{3,T}, u[celldofs(cell)]), mat)
# @assert norm(re) ≤ 10eps(T)
# u = ones(n) # passed: test with uniform displacement as well, should still be zero residual for membrane part (rigid body motion)
# membrane_residuals!(re, scv, x, reinterpret(Vec{3,T}, u[celldofs(cell)]), mat)
# @assert norm(re) ≤ 10eps(T)
# u .= 0; u[[1,4,7,10]] .= 0.1 # passed: small x-displacement to break rigid body motion, should give nonzero residual
# membrane_residuals!(re, scv, x, reinterpret(Vec{3,T}, u[celldofs(cell)]), mat)
# @assert norm(re) ≤ 10eps(T)


# # random field should sum to zero internally
# u = rand(n)
# membrane_residuals!(re, scv, x, reinterpret(Vec{3,T}, u[celldofs(cell)]), mat)
# total = zeros(3)
# for I in 1:length(scv.N)
#     total .+= re[3I-2:3I]
# end
# @assert all(total .≤ 10eps(T))

# # affine displacement should give zero residual for membrane part (rigid body motion + uniform stretch)
# u = vcat([xᵢ.*[0.01,0.02,0] for xᵢ in x]...)
# re .= 0
# membrane_residuals!(re, scv, x, reinterpret(Vec{3,T}, u[celldofs(cell)]), mat)
# total = zeros(3)
# for I in 1:length(scv.N)
#     total .+= re[3I-2:3I]
# end
# @assert all(total .≤ 10eps(T))


# # can we make something?
# ke .= 0
# membrane_tangent!(ke, scv, x, reinterpret(Vec{3,T}, u[celldofs(cell)]), mat)
# ke
# using LinearAlgebra
# eigenvals = eigvals(Matrix(ke))
# eigvecs = LinearAlgebra.eigvecs(Matrix(ke))
# n_rigid = count(x -> x ≤ 0, eigenvals)
# println("number of rigid body modes (zero eigenvalues): ", n_rigid)
# idx = sortperm(eigenvals)
# vals = eigenvals[idx]
# vecs = eigvecs[:, idx[n_rigid+1:end]]
# @show vecs

# try to assemble into global system
# K = allocate_matrix(dh)
# r = zeros(ndofs(dh))
# function assemble_membrane!(K, r, dh, scv, u, mat)
#     @assert length(u) == ndofs(dh) "u're not long enough mate!"
#     n = ndofs_per_cell(dh)
#     ke = zeros(n, n)
#     re  = zeros(n)
#     assembler = start_assemble(K, r)
#     for cell in CellIterator(dh)
#         x = getcoordinates(cell)
#         fill!(ke, 0.0); fill!(re, 0.0)
#         reinit!(scv, cell) # prepares reference geometry
#         u_e = reinterpret(Vec{3,T}, u[celldofs(cell)])
#         membrane_tangent!(ke, scv, x, u_e, mat)
#         membrane_residuals!(re, scv, x, u_e, mat)
#         assemble!(assembler, celldofs(cell), ke, re)
#     end
# end
# assemble_membrane!(K, r, dh, scv, u, mat)

"""
PATCH TEST
"""
grid = plate_grid(4, 4)
dh = DofHandler(grid)
add!(dh, :u, ip^3)
close!(dh)
K = allocate_matrix(dh)
r = zeros(ndofs(dh))
u = zeros(ndofs(dh))
assemble_membrane!(K, r, dh, scv, u, mat)
@assert norm(r) ≤ 10eps(T) "residual should be zero for zero displacements"

# set some boundary conditions
addfacetset!(grid, "clamped",  x -> x[1] ≈ 0.0)
addfacetset!(grid, "traction", x -> x[1] ≈ 1.0)

# apply boundary conditions
dbc = ConstraintHandler(dh)
add!(dbc, Dirichlet(:u, getfacetset(dh.grid, "clamped"), x -> zero(x), [1,2,3]))
add!(dbc, Dirichlet(:u, getfacetset(dh.grid, "traction"), x -> [0.01], [1]))
close!(dbc)

apply!(K, r, dbc)
u_e = K \ r


# T = Float64
# for cell in CellIterator(dh)
#     reinit!(scv, cell) # prepares reference geometry
#     fill!(ke, 0.0); fill!(re, 0.0)
#     x = getcoordinates(cell)
#     u_e = reinterpret(Vec{3,T}, u[celldofs(cell)])
#     @show u_e
#     for qp in 1:getnquadpoints(scv)
#         # construct current geometry for this quadrature point (using current displacements u)
#         ξ = scv.qr.points[qp]
#         # get reference geometry for this quadrature point
#         A₁ = zero(Vec{3,T})
#         A₂ = zero(Vec{3,T})
#         a₁ = zero(Vec{3,T})
#         a₂ = zero(Vec{3,T})
#         for i in 1:getnbasefunctions(scv.ip_geo)
#             dNdξ = Ferrite.reference_shape_gradient(scv.ip_geo, ξ, i)
#             A₁  += x[i] * dNdξ[1]
#             A₂  += x[i] * dNdξ[2]
#             a₁  += (x[i] + u_e[i]) * dNdξ[1]
#             a₂  += (x[i] + u_e[i]) * dNdξ[2]
#         end
#         @show A₁ A₂
#         @show a₁ a₂
#         A_metric = SymmetricTensor{2,2}((dot(A₁,A₁),dot(A₁,A₂),dot(A₂,A₂)))
#         a_metric = SymmetricTensor{2,2}((dot(a₁,a₁),dot(a₁,a₂),dot(a₂,a₂)))
#         @show A_metric a_metric
#         E = 0.5 * (a_metric - A_metric)
#         @show E
#         N = FerriteShells.membrane_stress(mat, E)
#         C = FerriteShells.membrane_tangent(mat)
#         @show N C
#         @show C ⊡ E # \boxdot
#         dΩ = scv.detJdV[qp]
#         @show dΩ "area element"
#         for I in 1:length(scv.N)
#             ∂NI1,∂NI2 = scv.∇N[qp, I]
#             # assemble residuals for this node I
#             v = ∂NI1 * (N[1,1]*a₁ + N[1,2]*a₂) +
#                 ∂NI2 * (N[2,1]*a₁ + N[2,2]*a₂)
#             # assemble into residual
#             re[3I-2:3I] .+= v * dΩ # checked, should be correct
#             for J in 1:length(scv.N)
#                 ∂NJ1,∂NJ2 = scv.∇N[qp, J]
#                 # geometric term
#                 geo_scalar = ∂NI1*(N[1,1]*∂NJ1 + N[1,2]*∂NJ2) +
#                              ∂NI2*(N[2,1]*∂NJ1 + N[2,2]*∂NJ2)
#                 Kgeo = geo_scalar * one(SymmetricTensor{2,3})
#                 @show geo_scalar Kgeo
#                 # # compute symmetric strain variation directions
#                 # B11 = ∂NJ1 * a₁
#                 # B22 = ∂NJ2 * a₂
#                 # B12 = 0.5*(∂NJ1*a₂ + ∂NJ2*a₁)
#                 # B = SymmetricTensor{2,2,eltype(a₁)}((B11, B12, B22))

#                 # resulting vector in ℝ³
#                 # δE11 = ∂NJ1 * dot(a₁, a₁)
#                 # δE22 = ∂NJ2 * dot(a₂, a₂)
#                 # δE12 = 0.5*(∂NJ1*dot(a₂,a₁) + ∂NJ2*dot(a₁,a₂))
#                 # δE = SymmetricTensor{2,2,eltype(a₁)}((δE11, δE12, δE22))
#                 # @show δE
#                 # # δN = C : δE double contraction
#                 # δN = C ⊡ δE
#                 # vector Sα = δN_{αβ} aβ
#                 # Kmat = ∂NI1 * (a₁ ⊗ δN[1] + a₂ ⊗ δN[1]) +
#                 #        ∂NI2 * (a₁ ⊗ δN[2] + a₂ ⊗ δN[2])
#                 Kmat = zero(SymmetricTensor{2,3})
#                 H1 = SymmetricTensor{2,2}((∂NJ1, 0.5∂NJ2, 0))
#                 H2 = SymmetricTensor{2,2}((0, 0.5∂NJ1, ∂NJ2))
#                 D1 = C ⊡ H1
#                 D2 = C ⊡ H2
#                 for (α,∂NIα) in enumerate((∂NI1, ∂NI2))
#                     for (β,aβ) in enumerate((a₁, a₂))
#                         # vector result from scalar contraction
#                         v = D1[α,β]*a₁ + D2[α,β]*a₂
#                         Kmat += ∂NIα * (aβ ⊗ v)
#                     end
#                 end
#                 @show Kmat
#                 # assemble
#                 ke[3I-2:3I, 3J-2:3J] .+= (Kgeo + Kmat) * dΩ
#             end
#         end
#     end
#     # assemble!(assembler, celldofs(cell), ke)
# end
# re
# using LinearAlgebra
# eigenvals = eigvals(Matrix(K))
# eigvecs = LinearAlgebra.eigvecs(Matrix(K))
# n_rigid = count(x -> x ≤ 0, eigenvals)
# println("number of rigid body modes (zero eigenvalues): ", n_rigid)
# idx = sortperm(eigenvals)
# vals = eigenvals[idx]
# vecs = eigvecs[:, idx[n_rigid+1:end]]


# traction in N/dm/thickness; right edge height = 60 - 44 = 16 → total force = 1 N
# traction = Vec{3}((0.0, 1/16*t, 0.0))

# assemble and solve
# Ke = allocate_matrix(dh)
# f  = zeros(ndofs(dh))
# assemble_shell!(Ke, dh, scv_mb, scv_s, E, ν, t)
# assemble_traction_force!(f, dh, getfacetset(grid, "traction"), traction)

# apply!(Ke, f, dbc)
# @time u = Ke \ f

# extract solution at point
# ph     = PointEvalHandler(grid, [Vec{3}((48.0, 60.0, 0.0))])
# u_eval = first(evaluate_at_points(ph, dh, u, :u))

# integrate edge traction force into f
# DOF ordering assumed: :u field first (3 DOFs per node, interleaved u,v,w),
# :θ field second.  This matches Ferrite's ordering when fields are added in that order.
# function assemble_traction_force!(f, dh, facetset, traction)
#     edge_local_nodes = Ferrite.reference_facets(RefQuadrilateral)  # ((1,2),(2,3),(3,4),(4,1))
#     n_dpc = ndofs_per_cell(dh)
#     fe    = zeros(n_dpc)
#     for fc in FacetIterator(dh, facetset)
#         x  = getcoordinates(fc)
#         fn = fc.current_facet_id             # local facet index: 1, 2, or 3
#         ia, ib = edge_local_nodes[fn]        # local node indices on this edge
#         edge_len = norm(x[ib] - x[ia])
#         fill!(fe, 0.0)
#         # 1-point midpoint quadrature: both edge nodes receive equal weight 0.5
#         # (exact for the linear shape functions used here)
#         for (node, N) in ((ia, 0.5), (ib, 0.5))
#             for c in 1:3    # u, v, w components
#                 fe[3(node-1)+c] += N * traction[c] * edge_len
#             end
#         end
#         f[celldofs(fc)] .+= fe
#     end
#     return nothing
# end