using FerriteShells,LinearAlgebra,Test,Random,ForwardDiff

# Shared geometry constants
const X_Q4_UNIT = [
    Vec{3}((0.0, 0.0, 0.0)), Vec{3}((1.0, 0.0, 0.0)),
    Vec{3}((1.0, 1.0, 0.0)), Vec{3}((0.0, 1.0, 0.0)),
]

const X_Q9_UNIT = [
    Vec{3}((0.0, 0.0, 0.0)), Vec{3}((1.0, 0.0, 0.0)),
    Vec{3}((1.0, 1.0, 0.0)), Vec{3}((0.0, 1.0, 0.0)),
    Vec{3}((0.5, 0.0, 0.0)), Vec{3}((1.0, 0.5, 0.0)),
    Vec{3}((0.5, 1.0, 0.0)), Vec{3}((0.0, 0.5, 0.0)),
    Vec{3}((0.5, 0.5, 0.0)),
]

# T3: right triangle corners; T6: corners + midpoints (Ferrite order: v1,v2,v3,mid12,mid23,mid31)
const X_T3_UNIT = [
    Vec{3}((0.0, 0.0, 0.0)), Vec{3}((1.0, 0.0, 0.0)), Vec{3}((0.0, 1.0, 0.0)),
]
const X_T6_UNIT = [
    Vec{3}((0.0, 0.0, 0.0)), Vec{3}((1.0, 0.0, 0.0)), Vec{3}((0.0, 1.0, 0.0)),
    Vec{3}((0.5, 0.0, 0.0)), Vec{3}((0.5, 0.5, 0.0)), Vec{3}((0.0, 0.5, 0.0)),
]

# Cell value constructors
make_q4_scv(; qr_order=1) = ShellCellValues(QuadratureRule{RefQuadrilateral}(qr_order),
                                             Lagrange{RefQuadrilateral,1}(), Lagrange{RefQuadrilateral,1}())
make_q9_scv(; qr_order=3) = ShellCellValues(QuadratureRule{RefQuadrilateral}(qr_order),
                                             Lagrange{RefQuadrilateral,2}(), Lagrange{RefQuadrilateral,2}())
make_t3_scv() = ShellCellValues(QuadratureRule{RefTriangle}(2),
                                Lagrange{RefTriangle,1}(), Lagrange{RefTriangle,1}())
make_t6_scv(; qr_order=4) = ShellCellValues(QuadratureRule{RefTriangle}(qr_order),
                                             Lagrange{RefTriangle,2}(), Lagrange{RefTriangle,2}())

# In-plane rotation about z
@inline R(θ) = Tensor{2,3}([cos(θ) -sin(θ) 0; sin(θ) cos(θ) 0; 0 0 1])

# KL membrane element strain energy
function element_strain_energy(scv, u_vec, mat)
    W = 0.0
    for qp in 1:getnquadpoints(scv)
        a₁, a₂, A_metric, a_metric = FerriteShells.kinematics(scv, qp, u_vec)
        E = 0.5 * (a_metric - A_metric)
        N = FerriteShells.contravariant_elasticity(mat, A_metric) ⊡ E
        W += 0.5 * (N ⊡ E) * scv.detJdV[qp]
    end
    W
end

# KL membrane helpers
function kl_residual(scv, u, mat)
    re = zeros(length(u)); membrane_residuals_KL!(re, scv, u, mat); re
end
function kl_tangent(scv, u, mat)
    ke = zeros(length(u), length(u)); membrane_tangent_KL!(ke, scv, u, mat); ke
end
function kl_fd_tangent(scv, u, mat; ε=1e-5)
    n = length(u); Kfd = zeros(n, n)
    for j in 1:n
        up = copy(u); up[j] += ε; um = copy(u); um[j] -= ε
        @views Kfd[:, j] = (kl_residual(scv, up, mat) .- kl_residual(scv, um, mat)) ./ (2ε)
    end
    Kfd
end

# KL bending helpers
function bending_residual(scv, u, mat)
    re = zeros(length(u)); bending_residuals_KL!(re, scv, u, mat); re
end
function bending_tangent(scv, u, mat)
    ke = zeros(length(u), length(u)); bending_tangent_KL!(ke, scv, u, mat); ke
end
function bending_fd_tangent(scv, u, mat; ε=1e-5)
    n = length(u); Kfd = zeros(n, n)
    for j in 1:n
        up = copy(u); up[j] += ε; um = copy(u); um[j] -= ε
        @views Kfd[:, j] = (bending_residual(scv, up, mat) .- bending_residual(scv, um, mat)) ./ (2ε)
    end
    Kfd
end

# RM helpers
function rm_residual(scv, u, mat)
    re = zeros(length(u))
    membrane_residuals_RM!(re, scv, u, mat)
    bending_residuals_RM_FD!(re, scv, u, mat)
    re
end
function rm_tangent(scv, u, mat)
    ke = zeros(length(u), length(u))
    membrane_tangent_RM!(ke, scv, u, mat)
    bending_tangent_RM_FD!(ke, scv, u, mat)
    ke
end
function rm_fd_tangent(scv, u, mat; ε=1e-5)
    n = length(u); Kfd = zeros(n, n)
    for j in 1:n
        up = copy(u); up[j] += ε; um = copy(u); um[j] -= ε
        @views Kfd[:, j] = (rm_residual(scv, up, mat) .- rm_residual(scv, um, mat)) ./ (2ε)
    end
    Kfd
end

# Mesh helpers
function patch_grid(; primitive=Quadrilateral)
    nodes = [
        Vec{3}(( 0.0,  0.0, 0.0)), Vec{3}((10.0,  0.0, 0.0)),
        Vec{3}((10.0, 10.0, 0.0)), Vec{3}(( 0.0, 10.0, 0.0)),
        Vec{3}(( 2.0,  2.0, 0.0)), Vec{3}(( 8.0,  3.0, 0.0)),
        Vec{3}(( 8.0,  7.0, 0.0)), Vec{3}(( 4.0,  7.0, 0.0)),
    ]
    cells = primitive == Triangle ?
        [(1,2,5),(2,6,5),(2,3,6),(3,7,6),(3,8,7),(3,4,8),(4,5,8),(4,1,5),(5,6,8),(6,7,8)] :
        [(1,2,6,5),(2,3,7,6),(3,4,8,7),(4,1,5,8),(5,6,7,8)]
    Grid([primitive(c) for c in cells], Node.(nodes))
end

function unit_square_mesh(n)
    nodes = [Vec{3}((i/n, j/n, 0.0)) for j in 0:n for i in 0:n]
    cells = [Quadrilateral((j*(n+1)+i+1, j*(n+1)+i+2, (j+1)*(n+1)+i+2, (j+1)*(n+1)+i+1))
             for j in 0:n-1 for i in 0:n-1]
    Grid(cells, Node.(nodes))
end

function make_u_exact(dh, f::Function)
    u = zeros(ndofs(dh))
    for cell in CellIterator(dh)
        coords = getcoordinates(cell); dofs = celldofs(cell)
        for i in eachindex(coords)
            val = f(coords[i])
            u[dofs[3i-2]] = val[1]; u[dofs[3i-1]] = val[2]; u[dofs[3i]] = val[3]
        end
    end
    u
end

function assemble_body_force!(r_ext, dh, scv, f_body)
    n_nodes = getnbasefunctions(scv.ip_shape)
    re = zeros(ndofs_per_cell(dh))
    for cell in CellIterator(dh)
        fill!(re, 0.0); reinit!(scv, cell)
        coords = getcoordinates(cell)
        for qp in 1:getnquadpoints(scv)
            ξ = scv.qr.points[qp]; dΩ = scv.detJdV[qp]
            x_qp = sum(Ferrite.reference_shape_value(scv.ip_shape, ξ, I) * coords[I] for I in 1:n_nodes)
            f = f_body(x_qp)
            for I in 1:n_nodes
                NI = Ferrite.reference_shape_value(scv.ip_shape, ξ, I)
                re[3I-2] += NI * f[1] * dΩ
                re[3I-1] += NI * f[2] * dΩ
            end
        end
        r_ext[celldofs(cell)] .+= re
    end
end

function l2_error(dh, scv, u_h, u_exact)
    n_nodes = getnbasefunctions(scv.ip_shape); err_sq = 0.0
    for cell in CellIterator(dh)
        reinit!(scv, cell); coords = getcoordinates(cell); u_e = u_h[celldofs(cell)]
        for qp in 1:getnquadpoints(scv)
            ξ = scv.qr.points[qp]; dΩ = scv.detJdV[qp]
            u_h_qp = sum(Ferrite.reference_shape_value(scv.ip_shape, ξ, I) *
                         Vec{3}((u_e[3I-2], u_e[3I-1], u_e[3I])) for I in 1:n_nodes)
            x_qp = sum(Ferrite.reference_shape_value(scv.ip_shape, ξ, I) * coords[I] for I in 1:n_nodes)
            diff = u_h_qp - u_exact(x_qp)
            err_sq += dot(diff, diff) * dΩ
        end
    end
    sqrt(err_sq)
end

function assemble_kl_residual!(r, dh, scv, u, mat)
    n = ndofs_per_cell(dh); re = zeros(n)
    for cell in CellIterator(dh)
        fill!(re, 0.0); reinit!(scv, cell)
        membrane_residuals_KL!(re, scv, u[celldofs(cell)], mat)
        r[celldofs(cell)] .+= re
    end
end

function assemble_kl_tangent!(K, r, dh, scv, u, mat)
    n = ndofs_per_cell(dh); ke = zeros(n, n); re = zeros(n)
    asm = start_assemble(K, r)
    for cell in CellIterator(dh)
        fill!(ke, 0.0); fill!(re, 0.0); reinit!(scv, cell)
        u_e = u[celldofs(cell)]
        membrane_tangent_KL!(ke, scv, u_e, mat)
        membrane_residuals_KL!(re, scv, u_e, mat)
        assemble!(asm, celldofs(cell), ke, re)
    end
end

include("test_kl.jl")
include("test_rm.jl")
include("test_mitc.jl")
include("test_mass.jl")
include("test_utils.jl")
include("test_benchmarks.jl")
