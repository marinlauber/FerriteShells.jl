using FerriteShells,BenchmarkTools

grid = generate_grid(Quadrilateral, (2, 2), [Vec{2}((0.0, 0.0)), Vec{2}((1.0, 0.0)), Vec{2}((1.0, 1.0)), Vec{2}((0.0, 1.0))]) |> shell_grid

# @inline function kinematics_strains(scv, qp, u_e::Vector{T}) where T
#     n_nodes = getnbasefunctions(scv.ip_shape)
#     Δa₁ = zero(Vec{3,T}); Δa₂ = zero(Vec{3,T})
#     for i in 1:n_nodes
#         ui  = Vec{3,T}((u_e[3i-2], u_e[3i-1], u_e[3i]))
#         Δa₁ += ui * scv.dNdξ[i, qp][1]
#         Δa₂ += ui * scv.dNdξ[i, qp][2]
#     end
#     a₁ = scv.A₁[qp] + Δa₁
#     a₂ = scv.A₂[qp] + Δa₂
#     e12 = 0.5 * (dot(scv.A₁[qp],Δa₂) + dot(scv.A₂[qp],Δa₁))
#     return a₁, a₂, SymmetricTensor{2,2,T}((dot(scv.A₁[qp],Δa₁), e12, dot(scv.A₂[qp],Δa₂)))
# end

ip = Lagrange{RefQuadrilateral, 1}()
qr = QuadratureRule{RefQuadrilateral}(2)
scv = ShellCellValues(qr, ip, ip; E=LinearStrain)

dh = DofHandler(grid)
add!(dh, :u, ip^5)
close!(dh)

f_int = zeros(ndofs(dh))
u     = zeros(ndofs(dh))

cell = first(CellIterator(dh))
ϵ = 0.01

# Build exact displacement vector (linear field, φ = 0).
# ε_xx, ε_yy, γ_xy = 1e-3, 2e-3, 5e-4
ε_xx, ε_yy, γ_xy = 1e-2, 0, 0
for cell in CellIterator(dh)
    coords = getcoordinates(cell); dofs = celldofs(cell)
    for (i, xi) in enumerate(coords)
        u[dofs[5i-4]] = ε_xx*xi[1] + 0.5γ_xy*xi[2]
        u[dofs[5i-3]] = 0.5γ_xy*xi[1] + ε_yy*xi[2]
    end
end

reinit!(scv, cell)
u_e = u[celldofs(cell)]
a₁, a₂, A_metric, a_metric = kinematics(scv, 1, u_e)

# # Green-Lagrange strain measure
GL = 0.5 * (a_metric - A_metric)
@show GL
# linear strain
Lin = kinematics_strains(scv, 1, u_e)[3]
@show Lin

@btime reinit!($scv, $cell) # 522.754 ns (0 allocations: 0 bytes)
@btime kinematics_strains($scv, $1, $u_e) # 9.182 ns (0 allocations: 0 bytes)
@btime kinematics($scv, $1, $u_e) # 12.534 ns (0 allocations: 0 bytes)