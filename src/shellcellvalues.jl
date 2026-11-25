
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
* `frames`: a [`NodeFrames`](@ref) object (default `nothing`). When given, the cell-based
    `reinit!(scv, cell)` / `reinit!(scv, cc)` entry points use the stored per-node frames
    automatically, so a kernel loop written as `reinit!(scv, cell)` cannot silently fall back
    to centroid frames. Pass `nothing` for centroid frames.
* `dofs`: the `SubDofHandler` (or a `DofHandler` with a single one) whose cells this object
    is used on (default `nothing`). The kernels take the element vector in Ferrite's own
    cell-dof order, `u[celldofs(cell)]`, and locate the displacement and rotation dofs in it
    through `Ferrite.dof_range(sdh, :u)` and `Ferrite.dof_range(sdh, :θ)`, read once here.
    With `nothing` they default to `1:3n` and `3n+1:5n`, the layout of
    `add!(dh, :u, ip^3); add!(dh, :θ, ip^2)` (and, for `:u` alone, of a Kirchhoff–Love
    `DofHandler`); pass `dofs` for any other field order or additional fields.

**Common methods:**
* [`reinit!`](@ref) computes the reference geometry (``A_1``, ``A_2``, ``G_3``, ``B``, ``\\cdots``) by differentiating the coordinate map using `ip_geo`.
* [`copy`](@ref copy(::ShellCellValues)) returns an independent object that shares the immutable
    rule / interpolations / frames but owns its own `reinit!` buffers — the sanctioned idiom for
    multi-threaded assembly is one `copy(scv)` per task (or per cell-chunk).
"""
ShellCellValues

struct ShellCellValues{QR, IPG, IPS, T<:AbstractFloat, M, F} <: AbstractCellValues
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
    B₀       :: Vector{SymmetricTensor{2, 2, T, 3}}  # reference director-gradient curvature, see reference_director_curvature!
    G₃_elem  :: Vector{Vec{3, T}}   # per-node frame (length n_shape) — updated each reinit!
    T₁_elem  :: Vector{Vec{3, T}}
    T₂_elem  :: Vector{Vec{3, T}}
    mitc     :: M  # Nothing, or an AbstractMITCData (e.g. MITC9Data) for locking-free shear
    frames   :: F  # Nothing, or a NodeFrames used by the cell-based reinit! entry points
    ru       :: UnitRange{Int}  # local positions of the :u dofs (3 per node) in the element vector
    rθ       :: UnitRange{Int}  # local positions of the :θ dofs (2 per node); empty without :θ
end

Ferrite.getnormal(scv::ShellCellValues, q::Int) = scv.G₃[q]
Ferrite.getdetJdV(scv::ShellCellValues, q::Int) = scv.detJdV[q]
Ferrite.getnquadpoints(scv::ShellCellValues) = getnquadpoints(scv.qr)
Ferrite.getnbasefunctions(scv::ShellCellValues) = getnbasefunctions(scv.ip_shape)
@propagate_inbounds Ferrite.getngeobasefunctions(scv::ShellCellValues) = getnbasefunctions(scv.ip_geo)

function ShellCellValues(qr::QuadratureRule, ip_geo::Interpolation, ip_shape::Interpolation; mitc=nothing, frames=nothing, dofs=nothing)
    n_qp    = length(qr.weights)
    n_shape = getnbasefunctions(ip_shape)
    T       = Float64
    ru, rθ  = shell_dof_ranges(dofs, n_shape)

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
    ShellCellValues{typeof(qr), typeof(ip_geo), typeof(ip_shape), T, typeof(m), typeof(frames)}(
        qr, ip_geo, ip_shape,
        N, dNdξ, d2Ndξ2, zeros(T, n_qp),
        fill(zero(Vec{3, T}), n_qp), fill(zero(Vec{3, T}), n_qp),
        fill(zero(Vec{3, T}), n_qp), fill(zero(Vec{3, T}), n_qp),
        fill(zero(Vec{3, T}), n_qp), fill(zero(SymmetricTensor{2, 2, T, 3}), n_qp),
        fill(zero(Vec{3, T}), n_qp), fill(zero(Vec{3, T}), n_qp),
        fill(zero(Vec{3, T}), n_qp), fill(zero(SymmetricTensor{2, 2, T, 3}), n_qp),
        fill(zero(SymmetricTensor{2, 2, T, 3}), n_qp),
        fill(zero(Vec{3, T}), n_shape), fill(zero(Vec{3, T}), n_shape),
        fill(zero(Vec{3, T}), n_shape), m, frames, ru, rθ
    )
end

# Local positions of the `:u` and `:θ` dofs in the element vector, read by name from the
# SubDofHandler, or the default two-field layout (`:u` as ip^3 first, `:θ` as ip^2).
shell_dof_ranges(::Nothing, n::Int) = 1:3n, 3n+1:5n
shell_dof_ranges(dh::DofHandler, n::Int) = shell_dof_ranges(only(dh.subdofhandlers), n)
function shell_dof_ranges(sdh::SubDofHandler, n::Int)
    names = Ferrite.getfieldnames(sdh)
    :u in names || throw(ArgumentError("the SubDofHandler has no :u field"))
    ru = Ferrite.dof_range(sdh, :u)
    rθ = :θ in names ? Ferrite.dof_range(sdh, :θ) : (1:0)
    length(ru) == 3n || throw(ArgumentError(
        ":u has $(length(ru)) dofs per cell, expected $(3n) (ip^3 with $n shape functions)"))
    (isempty(rθ) || length(rθ) == 2n) || throw(ArgumentError(
        ":θ has $(length(rθ)) dofs per cell, expected $(2n) (ip^2 with $n shape functions)"))
    return ru, rθ
end

"""
    Base.copy(scv::ShellCellValues)

An independent `ShellCellValues` sharing the immutable rule/interpolations/frames but
with its own reinit! buffers (including the MITC data): the threaded-assembly idiom is
one `copy(scv)` per task.
"""
function Base.copy(scv::ShellCellValues)
    typeof(scv)(
        scv.qr, scv.ip_geo, scv.ip_shape,
        copy(scv.N), copy(scv.dNdξ), copy(scv.d2Ndξ2), copy(scv.detJdV),
        copy(scv.A₁), copy(scv.A₂), copy(scv.A₁₁), copy(scv.A₁₂), copy(scv.A₂₂),
        copy(scv.A_metric), copy(scv.G₃), copy(scv.T₁), copy(scv.T₂), copy(scv.B),
        copy(scv.B₀), copy(scv.G₃_elem), copy(scv.T₁_elem), copy(scv.T₂_elem),
        copy(scv.mitc), scv.frames, scv.ru, scv.rθ
    )
end

"""
    function_value(scv, qp, u_e)

Interpolate the displacement field at quadrature point `qp` from the element vector
`u_e = u[celldofs(cell)]`. The displacement dofs are read at `scv.ru`, the `:u` dof range
(see the `dofs` keyword of [`ShellCellValues`](@ref)); rotation dofs are ignored.
"""
@inline function Ferrite.function_value(scv::ShellCellValues, qp::Int, u_e::AbstractVector{T}) where T
    n_nodes = getnbasefunctions(scv.ip_shape)
    ru  = scv.ru
    val = zero(Vec{3, T})
    @inbounds for I in 1:n_nodes
        val += scv.N[I, qp] * Vec{3}(@view u_e[ru[3I-2:3I]])
    end
    val
end

"""
    function_gradient(scv, qp, u_e)

Parametric displacement gradient ``\\partial u/\\partial \\xi_\\alpha`` at quadrature point `qp`
(third column zero), with the displacement dofs of `u_e` read at `scv.ru`.
"""
@inline function Ferrite.function_gradient(scv::ShellCellValues, qp::Int, u_e::AbstractVector{T}) where T
    n_nodes = getnbasefunctions(scv.ip_shape)
    ru  = scv.ru
    val = zero(Tensor{2,3,T})
    @inbounds for I in 1:n_nodes
        N = scv.dNdξ[I, qp]
        Nvec = Vec{3,T}((N[1], N[2], 0))
        val += Vec{3}(@view u_e[ru[3I-2:3I]]) ⊗ Nvec
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
reinit!(::ShellCellValues, x::AbstractVector{<:Vec{3}})

# Cell-based entry points route through the frames stored at construction (if any),
# so kernels written as `reinit!(scv, cell)` cannot silently drop NodeFrames.
reinit!(scv::ShellCellValues, cell) = _reinit_cell!(scv, scv.frames, cell)
reinit!(scv::ShellCellValues, cc::CellCache) = _reinit_cell!(scv, scv.frames, cc)
_reinit_cell!(scv::ShellCellValues, ::Nothing, cell) = reinit!(scv, getcoordinates(cell))
# frames::NodeFrames — untyped; NodeFrames is defined in nodeframe.jl.
_reinit_cell!(scv::ShellCellValues, frames, cell) = reinit!(scv, getcoordinates(cell), frames, getnodes(cell))

# Per-QP reference geometry: covariant tangents/curvature, metric, patch normal frame,
# and detJdV. Independent of which element-director frame (centroid vs. NodeFrames) is
# used downstream, so both `reinit!` paths share this instead of each recomputing it.
function reinit_geometry!(scv::ShellCellValues, x::AbstractVector{<:Vec{3}})
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
end

function reinit!(scv::ShellCellValues, x::AbstractVector{<:Vec{3}})
    reinit_geometry!(scv, x)
    n_geo = getnbasefunctions(scv.ip_geo)
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
    n_shape = getnbasefunctions(scv.ip_shape)
    for I in 1:n_shape
        scv.G₃_elem[I] = G₃_c; scv.T₁_elem[I] = T₁_c; scv.T₂_elem[I] = T₂_c
    end
    reference_director_curvature!(scv)
    reinit!(scv.mitc, scv.ip_geo, x, scv.G₃_elem, scv.T₁_elem, scv.T₂_elem)
end

# Reference director-gradient curvature B₀_αβ = ½(A_α·d₀,β + A_β·d₀,α), with
# d₀ = Σ N_I G₃_elem[I] the initial director field interpolated from the frames the
# element actually rotates about. The bending measure of the RM kernels is the
# *change* κ = ½(a_α·d,β + a_β·d,α) − B₀, which vanishes identically at u = 0 for
# any geometry and any frame choice. Subtracting the patch curvature B instead — the
# earlier behaviour — leaves a spurious reference bending strain κ(0) = B₀ − B on
# curved or warped elements, and it does not converge away: with per-node frames
# (`NodeFrames`) the Scordelis-Lo roof stalls at 9.3% error under refinement, and with
# centroid frames the answer drifts past the reference instead of settling on it.
# On flat elements with centroid frames B₀ = B = 0 and nothing changes.
function reference_director_curvature!(scv::ShellCellValues)
    n_shape = getnbasefunctions(scv.ip_shape)
    for q in 1:getnquadpoints(scv)
        d0_1 = zero(Vec{3, Float64}); d0_2 = zero(Vec{3, Float64})
        for I in 1:n_shape
            dN = scv.dNdξ[I, q]
            d0_1 += dN[1] * scv.G₃_elem[I]
            d0_2 += dN[2] * scv.G₃_elem[I]
        end
        A₁, A₂ = scv.A₁[q], scv.A₂[q]
        scv.B₀[q] = SymmetricTensor{2, 2, Float64}(
            (dot(A₁, d0_1), 0.5 * (dot(A₁, d0_2) + dot(A₂, d0_1)), dot(A₂, d0_2))
        )
    end
    return scv
end

# compute the centroid coordinates for different element topologies
@inline reference_centroid(::Interpolation{RefQuadrilateral}) = Vec{2}((0.0, 0.0))
@inline reference_centroid(::Interpolation{RefTriangle})      = Vec{2}((1/3, 1/3))
