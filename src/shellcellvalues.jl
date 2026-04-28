
"""
    ShellCellValues(geom_interpol::Interpolation, func_interpol::Interpolation, quad_rule::AbstractQuadratureRule)

A `ShellCellValues` object stores precomputed shape function data and reference geometry for a shell element.
Works with `Vec{3}` node coordinates — no manual 2D projection required.

Shape function data (`N`, `dNdξ`, `d2Ndξ2`) are computed once at construction time from `ip_shape` and `qr`.
The reference coordinates in the physical space then computed on the fly using the geometric interpolation `ip_geo`
``x(\\xi) = \\sum N_{i}^\\text{geo}(\\xi) x_{i}`` and the solution field are interpolated via the shape
functions `ip_shape` ``u(\\xi) = \\sum N_{i}^\\text{shape}(\\xi) u_{i}``.

**Arguments:**
* `geom_interpol`: an instance of an `Interpolation` which is used to interpolate the geometry.
    By default linear Lagrange interpolation is used.
* `func_interpol`: an instance of an `Interpolation` used to interpolate the approximated function
* `quad_rule`: an instance of a `AbstractQuadratureRule`

**Keyword arguments:** The following keyword arguments are experimental and may change in future minor releases
* `mitc`:  an instant of [`MITC`](@ref) to specify the shear treatment used in the element (default `NoMITC`)
* `E` : an instance of `AbstractStrainMeasure` to specify the strain measure used in the element (default `GreenLagrangeStrain`)

**Common methods:**
* [`reinit!`](@ref) computes the reference geometry (``A_1``, ``A_2``, ``G_3``, ``B``, ``\\cdots``) by differentiating the coordinate map using `ip_geo`.
"""
ShellCellValues

struct ShellCellValues{QR, IPG, IPS, T<:AbstractFloat, E<:AbstractStrainMeasure, M} <: AbstractCellValues
    qr       :: QR
    ip_geo   :: IPG
    ip_shape :: IPS
    N        :: Matrix{T}
    dNdξ     :: Matrix{Vec{2, T}}
    d2Ndξ2   :: Matrix{SymmetricTensor{2, 2, T, 3}}
    detJdV   :: Vector{T}
    A₁       :: Vector{Vec{3, T}}
    A₂       :: Vector{Vec{3, T}}
    A₁₁      :: Vector{Vec{3, T}}
    A₁₂      :: Vector{Vec{3, T}}
    A₂₂      :: Vector{Vec{3, T}}
    A_metric :: Vector{SymmetricTensor{2, 2, T, 3}}
    G₃       :: Vector{Vec{3, T}}
    T₁       :: Vector{Vec{3, T}}
    T₂       :: Vector{Vec{3, T}}
    B        :: Vector{SymmetricTensor{2, 2, T, 3}}
    G₃_elem  :: Vector{Vec{3, T}}   # element-centroid frame (length 1) — shared by all QPs
    T₁_elem  :: Vector{Vec{3, T}}
    T₂_elem  :: Vector{Vec{3, T}}
    mitc     :: M  # Nothing, or an AbstractMITCData (e.g. MITC9Data) for locking-free shear
end

Ferrite.getnormal(scv::ShellCellValues, q::Int) = scv.G₃[q]
Ferrite.getdetJdV(scv::ShellCellValues, q::Int) = scv.detJdV[q]
Ferrite.getnquadpoints(scv::ShellCellValues) = getnquadpoints(scv.qr)
Ferrite.getnbasefunctions(scv::ShellCellValues) = getnbasefunctions(scv.ip_shape)
@propagate_inbounds Ferrite.getngeobasefunctions(scv::ShellCellValues) = getnbasefunctions(scv.ip_geo)

function ShellCellValues(qr::QuadratureRule, ip_geo::Interpolation, ip_shape::Interpolation; E=GreenLagrangeStrain, mitc=nothing)
    n_qp    = length(qr.weights)
    n_shape = getnbasefunctions(ip_shape)
    T       = Float64

    N      = zeros(T, n_shape, n_qp)
    dNdξ   = Matrix{Vec{2, T}}(undef, n_shape, n_qp)
    d2Ndξ2 = Matrix{SymmetricTensor{2, 2, T, 3}}(undef, n_shape, n_qp)
    for q in 1:n_qp
        ξ = qr.points[q]
        for I in 1:n_shape
            d2N, dN, Nval = Ferrite.reference_shape_hessian_gradient_and_value(ip_shape, ξ, I)
            N[I, q]      = Nval
            dNdξ[I, q]   = dN
            d2Ndξ2[I, q] = SymmetricTensor{2, 2, T}((d2N[1,1], d2N[1,2], d2N[2,2]))
        end
    end

    m = isnothing(mitc) ? NoMITC() : mitc(ip_shape, qr)
    ShellCellValues{typeof(qr), typeof(ip_geo), typeof(ip_shape), T, E, typeof(m)}(
        qr, ip_geo, ip_shape,
        N, dNdξ, d2Ndξ2, zeros(T, n_qp),
        fill(zero(Vec{3, T}), n_qp), fill(zero(Vec{3, T}), n_qp),
        fill(zero(Vec{3, T}), n_qp), fill(zero(Vec{3, T}), n_qp),
        fill(zero(Vec{3, T}), n_qp), fill(zero(SymmetricTensor{2, 2, T, 3}), n_qp),
        fill(zero(Vec{3, T}), n_qp), fill(zero(Vec{3, T}), n_qp),
        fill(zero(Vec{3, T}), n_qp), fill(zero(SymmetricTensor{2, 2, T, 3}), n_qp),
        fill(zero(Vec{3, T}), 1), fill(zero(Vec{3, T}), 1),
        fill(zero(Vec{3, T}), 1), m
    )
end

"""
    function_value(scv, qp, u_e)

Interpolate the displacement field at quadrature point `qp` from a flat DOF vector `u_e`.
Works for both KL (3 DOFs/node: ``[u_1,u_2,u_3,\\cdots]``) and RM (5 DOFs/node: ``[u_1,u_2,u_3,\\varphi_1,\\varphi_2,\\cdots]``).
The DOF stride is inferred from `length(u_e) ÷ n_nodes`; only the first 3 DOFs of each node
(the displacement components) are used.
"""
@inline function Ferrite.function_value(scv::ShellCellValues, qp::Int, u_e::AbstractVector{T}) where T
    n_nodes = getnbasefunctions(scv.ip_shape)
    stride  = length(u_e) ÷ n_nodes
    val = zero(Vec{3, T})
    @inbounds for I in 1:n_nodes
        o = stride * (I - 1)
        val += scv.N[I, qp] * Vec{3, T}((u_e[o + 1], u_e[o + 2], u_e[o + 3]))
    end
    val
end

@inline function Ferrite.function_gradient(scv::ShellCellValues, qp::Int, u_e::AbstractVector{T}) where T
    n_nodes = getnbasefunctions(scv.ip_shape)
    stride  = length(u_e) ÷ n_nodes
    val = zero(Tensor{2,3,T})
    @inbounds for I in 1:n_nodes
        o = stride * (I - 1)
        N = scv.dNdξ[I, qp]
        Nvec = Vec{3,T}((N[1], N[2], 0))
        val += Vec{3, T}((u_e[o + 1], u_e[o + 2], u_e[o + 3])) ⊗ Nvec
    end
    val
end

# should overwrite ``geometric_value(::ShellCellValues, ::Any...)``
@inline function Ferrite.spatial_coordinate(scv::ShellCellValues, qp::Int, x::AbstractVector{<:Vec})
    n_nodes = getnbasefunctions(scv.ip_geo)
    ξ = scv.qr.points[qp]
    val = zero(Vec{3, eltype(ξ)})
    @inbounds for I in 1:n_nodes
        val += Ferrite.reference_shape_value(scv.ip_geo, ξ, I) * x[I]
    end
    val
end

"""
    reinit!(scv::ShellCellValues, x::AbstractVector)
    reinit!(scv::ShellCellValues, cc::CellCache)
    reinit!(scv::ShellCellValues, cell::AbstractCell)

Update the `ShellCellValues` object for a cell with cell coordinates `x`.
The derivatives of the shape functions, and the new integration weights are computed.

The reference surface measures such as the covariant basis are recomputed and stored per
quadrature point.

**Note:**
For `ShellCellValues` where a shear treatment has been specified, the `MITC` data is also `reinit!`.
"""
reinit!

reinit!(scv::ShellCellValues, cell) = reinit!(scv, getcoordinates(cell))
reinit!(scv::ShellCellValues, cc::CellCache) = reinit!(scv, getcoordinates(cc))
function reinit!(scv::ShellCellValues, x::AbstractVector{<:Vec{3}})
    n_geo = getnbasefunctions(scv.ip_geo)
    for q in eachindex(scv.qr.weights)
        ξ = scv.qr.points[q]
        A₁  = zero(Vec{3,Float64}); A₂  = zero(Vec{3,Float64})
        A₁₁ = zero(Vec{3,Float64}); A₁₂ = zero(Vec{3,Float64}); A₂₂ = zero(Vec{3,Float64})
        for i in 1:n_geo
            d2N, dN, _ = Ferrite.reference_shape_hessian_gradient_and_value(scv.ip_geo, ξ, i)
            A₁  += x[i] * dN[1];    A₂  += x[i] * dN[2]
            A₁₁ += x[i] * d2N[1,1]; A₁₂ += x[i] * d2N[1,2]; A₂₂ += x[i] * d2N[2,2]
        end
        n_vec = A₁ × A₂
        area  = norm(n_vec)
        G₃    = n_vec / area
        T₁    = A₁ / norm(A₁)
        T₂    = (G₃ × T₁) / norm(G₃ × T₁)

        scv.detJdV[q]   = area * scv.qr.weights[q]
        scv.A₁[q]       = A₁;  scv.A₂[q]  = A₂
        scv.A₁₁[q]      = A₁₁; scv.A₁₂[q] = A₁₂; scv.A₂₂[q] = A₂₂
        scv.A_metric[q] = SymmetricTensor{2,2,Float64}((dot(A₁,A₁), dot(A₁,A₂), dot(A₂,A₂)))
        scv.G₃[q]       = G₃;  scv.T₁[q]  = T₁;  scv.T₂[q]  = T₂
        scv.B[q]        = SymmetricTensor{2,2,Float64}((dot(A₁₁,G₃), dot(A₁₂,G₃), dot(A₂₂,G₃)))
    end
    # Centroid frame — single consistent director frame for the whole element.
    # T₁_c is chosen by Gram-Schmidt projection of a global reference vector (ê_x) onto
    # the element tangent plane. For flat shells this gives T₁_c = ê_x and T₂_c = ê_y
    # for every element regardless of shape, eliminating both within-element (QP-to-QP)
    # and inter-element (shared-node) frame inconsistency.
    ξ_c  = reference_centroid(scv.ip_geo)
    A₁_c = zero(Vec{3,Float64}); A₂_c = zero(Vec{3,Float64})
    for i in 1:n_geo
        dN, _ = Ferrite.reference_shape_gradient_and_value(scv.ip_geo, ξ_c, i)
        A₁_c += x[i] * dN[1]; A₂_c += x[i] * dN[2]
    end
    n_c  = A₁_c × A₂_c
    G₃_c = n_c / norm(n_c)
    ref  = abs(G₃_c[1]) < 0.9 ? Vec{3}((1.,0.,0.)) : Vec{3}((0.,1.,0.))
    t₁   = ref - (ref ⋅ G₃_c) * G₃_c
    T₁_c = t₁ / norm(t₁)
    T₂_c = G₃_c × T₁_c
    scv.G₃_elem[1] = G₃_c; scv.T₁_elem[1] = T₁_c; scv.T₂_elem[1] = T₂_c
    reinit!(scv.mitc, scv.ip_geo, x, G₃_c, T₁_c, T₂_c)
end

# compute the centroid coordinates for different element topologies
@inline reference_centroid(::Interpolation{RefQuadrilateral}) = Vec{2}((0.0, 0.0))
@inline reference_centroid(::Interpolation{RefTriangle})      = Vec{2}((1/3, 1/3))
