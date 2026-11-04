# Prototype: geometrically nonlinear Reissner-Mindlin shell on Ferrite.jl PR #1544, Design 3
# (local-frame derivatives in CellValues). Fields: displacement u (ip^3) and a global rotation
# vector ω (ip^3). Nodal director interpolation d = Σ N_a R(ω_a)·D_a; the drilling rotation is
# removed by the linear constraints ω_a·D_a = 0 (see `drilling_constraints!`).
using Ferrite, Tensors, LinearAlgebra

# Rodrigues coefficients cosθ, sinθ/θ, (1-cosθ)/θ² as functions of θ² (smooth and AD-safe at 0)
@inline function rodrigues_coeffs(θ²)
    if θ² < 1e-6
        c = 1 - θ²/2 + θ²^2/24 - θ²^3/720
        s = 1 - θ²/6 + θ²^2/120 - θ²^3/5040
        h = 1/2 - θ²/24 + θ²^2/720 - θ²^3/40320
    else
        θ = sqrt(θ²)
        c, s, h = cos(θ), sin(θ)/θ, (1 - cos(θ))/θ²
    end
    return c, s, h
end
@inline function rotate(ω::Vec{3}, D::Vec{3})
    c, s, h = rodrigues_coeffs(ω ⋅ ω)
    return c*D + s*(ω × D) + h*(ω ⋅ D)*ω
end

function plane_stress(E, ν)
    I = one(SymmetricTensor{2, 2})
    return E / (1 - ν^2) * ((1 - ν) * one(SymmetricTensor{4, 2}) + ν * I ⊗ I)
end

struct RMShell{CV, G}
    cv::CV
    A::SymmetricTensor{4, 2, Float64, 9}   # membrane stiffness
    Dm::SymmetricTensor{4, 2, Float64, 9}  # bending stiffness
    As::Tensor{2, 2, Float64, 4}           # transverse shear stiffness
    # test-function strains and their gradients in a quadrature point
    δε::Vector{SymmetricTensor{2, 2, Float64, 3}}
    δκ::Vector{SymmetricTensor{2, 2, Float64, 3}}
    δγ::Vector{Vec{2, Float64}}
    δF::Vector{G}                 # ∇ₛN (u dofs) or δ(∇ₛd) (ω dofs)
    δd::Vector{Vec{3, Float64}}   # 0 (u dofs) or δd (ω dofs)
    # nodal directors for the current cell
    D::Vector{Vec{3, Float64}}            # reference director D_a (set before calling element!)
    Λ1::Vector{Tensor{2, 3, Float64, 9}}  # ∂d_a/∂ω_a
    Λ2::Vector{Tensor{3, 3, Float64, 27}} # ∂²d_a/∂ω_a²
    de::Vector{Float64}                   # d_a, flattened like the ω dofs
    De::Vector{Float64}                   # D_a, flattened like the ω dofs
end

function RMShell(ip, ip_geo, qr; E, ν, t, κs = 5/6)
    cv = CellValues(qr, (u = ip^3, ω = ip^3), ip_geo^3; update_local_gradients = true)
    C = plane_stress(E, ν)
    n, na = 2getnbasefunctions(cv.u), getnbasefunctions(ip)
    G = typeof(shape_local_gradient(cv.u, 1, 1))
    return RMShell(cv, t * C, t^3 / 12 * C, κs * t * E / (2(1 + ν)) * one(Tensor{2, 2}),
        zeros(SymmetricTensor{2, 2, Float64, 3}, n), zeros(SymmetricTensor{2, 2, Float64, 3}, n),
        zeros(Vec{2, Float64}, n), zeros(G, n), zeros(Vec{3, Float64}, n),
        zeros(Vec{3, Float64}, na), zeros(Tensor{2, 3, Float64, 9}, na), zeros(Tensor{3, 3, Float64, 27}, na),
        zeros(3na), zeros(3na))
end

# Residual `re` and tangent `ke` for element dofs ordered [u; ω] (Ferrite's field-blocked layout).
function element!(ke, re, s::RMShell, ue)
    cv = s.cv
    nu, nω = getnbasefunctions(cv.u), getnbasefunctions(cv.ω)
    ru, rω = 1:nu, (nu + 1):(nu + nω)
    # Nodal directors d_a = R(ω_a)·D_a and their first and second derivatives wrt. ω_a
    for a in eachindex(s.D)
        ω, D = Vec{3}(k -> ue[nu + 3(a - 1) + k]), s.D[a]
        s.Λ2[a], s.Λ1[a] = Tensors.gradient(w -> Tensors.gradient(v -> rotate(v, D), w), ω, :all)
        d = rotate(ω, D)
        for k in 1:3
            s.de[3(a - 1) + k], s.De[3(a - 1) + k] = d[k], D[k]
        end
    end
    for qp in 1:getnquadpoints(cv)
        E, dA = Ferrite.getlocalframe(cv, qp), getdetJdV(cv, qp)
        # Kinematics: F = [a₁ a₂], director d and its gradient, reference values D₀, ∇D₀
        F = E + function_local_gradient(cv.u, qp, ue, ru)
        d, ∇d = function_value(cv.ω, qp, s.de), function_local_gradient(cv.ω, qp, s.de)
        D₀, ∇D₀ = function_value(cv.ω, qp, s.De), function_local_gradient(cv.ω, qp, s.De)
        ε = (symmetric(F' ⋅ F) - one(SymmetricTensor{2, 2})) / 2
        κ = symmetric(F' ⋅ ∇d) - symmetric(E' ⋅ ∇D₀)
        γ = F' ⋅ d - E' ⋅ D₀
        N, M, Q = s.A ⊡ ε, s.Dm ⊡ κ, s.As ⋅ γ
        # Strains of the test functions
        for i in ru
            δF = shape_local_gradient(cv.u, qp, i)
            s.δε[i], s.δκ[i], s.δγ[i] = symmetric(F' ⋅ δF), symmetric(δF' ⋅ ∇d), δF' ⋅ d
            s.δF[i], s.δd[i] = δF, zero(Vec{3})
        end
        for (j, i) in pairs(rω)
            Λ1 = s.Λ1[(j - 1) ÷ 3 + 1]
            δd, δ∇d = Λ1 ⋅ shape_value(cv.ω, qp, j), Λ1 ⋅ shape_local_gradient(cv.ω, qp, j)
            s.δε[i], s.δκ[i], s.δγ[i] = zero(SymmetricTensor{2, 2}), symmetric(F' ⋅ δ∇d), F' ⋅ δd
            s.δF[i], s.δd[i] = δ∇d, δd
        end
        # Residual and material tangent (upper triangle; mirrored after the quadrature loop)
        for i in 1:(nu + nω)
            re[i] += (N ⊡ s.δε[i] + M ⊡ s.δκ[i] + Q ⋅ s.δγ[i]) * dA
            Aδε, Dδκ, Asδγ = s.A ⊡ s.δε[i], s.Dm ⊡ s.δκ[i], s.As' ⋅ s.δγ[i]
            for j in i:(nu + nω)
                ke[i, j] += (Aδε ⊡ s.δε[j] + Dδκ ⊡ s.δκ[j] + Asδγ ⋅ s.δγ[j]) * dA
            end
        end
        # Geometric tangent: uu (membrane), uω (bending + shear), ωω (same node only)
        for i in ru
            for j in i:nu
                ke[i, j] += (N ⊡ (s.δF[i]' ⋅ s.δF[j])) * dA
            end
            for j in rω
                ke[i, j] += (M ⊡ (s.δF[i]' ⋅ s.δF[j]) + Q ⋅ (s.δF[i]' ⋅ s.δd[j])) * dA
            end
        end
        for a in eachindex(s.D)
            i = 3(a - 1) + 1
            Na, ∇Na = shape_value(cv.ω, qp, i)[1], Vec{2}(shape_local_gradient(cv.ω, qp, i)[1, :])
            W = (F ⋅ (M ⋅ ∇Na + Na * Q)) ⋅ s.Λ2[a]
            for k in 1:3, l in k:3
                ke[nu + i - 1 + k, nu + i - 1 + l] += W[k, l] * dA
            end
        end
    end
    for j in axes(ke, 2), i in (j + 1):size(ke, 1)
        ke[i, j] = ke[j, i]
    end
    return ke, re
end

# Area-weighted averaged nodal normals (same construction as FerriteShells.NodeFrames)
function nodal_normals(grid, ip_geo)
    n = fill(zero(Vec{3}), getnnodes(grid))
    ξc = Vec{2}((0.0, 0.0))
    for cell in CellIterator(grid)
        x = getcoordinates(cell)
        J = sum(x[i] ⊗ Ferrite.reference_shape_gradient(ip_geo, ξc, i) for i in eachindex(x))
        m = J[:, 1] × J[:, 2]
        for nid in getnodes(cell)
            n[nid] += m
        end
    end
    return normalize.(n)
end

# ω_a·D_a = 0 for every node without prescribed rotations: eliminate the component with the largest |D_k|
function drilling_constraints!(ch, dh, Dnodes)
    prescribed = Set(ch.prescribed_dofs)
    done = falses(length(Dnodes))
    rω = dof_range(dh, :ω)
    for cell in CellIterator(dh)
        cd = celldofs(cell)
        for (a, nid) in enumerate(getnodes(cell))
            done[nid] && continue
            done[nid] = true
            dofs = cd[rω[3(a - 1) .+ (1:3)]]
            any(in(prescribed), dofs) && continue
            D = Dnodes[nid]
            k = argmax(abs.(D))
            add!(ch, AffineConstraint(dofs[k], [dofs[j] => -D[j] / D[k] for j in 1:3 if j != k], 0.0))
        end
    end
    return ch
end

function assemble_global!(K, r, dh, s::RMShell, Dnodes, u)
    n = ndofs_per_cell(dh)
    ke, re = zeros(n, n), zeros(n)
    asm = start_assemble(K, r)
    for cell in CellIterator(dh)
        fill!(ke, 0.0); fill!(re, 0.0)
        reinit!(s.cv, getcoordinates(cell))
        s.D .= view(Dnodes, getnodes(cell))
        element!(ke, re, s, u[celldofs(cell)])
        assemble!(asm, celldofs(cell), ke, re)
    end
    return K, r
end
