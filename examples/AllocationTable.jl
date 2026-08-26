# Prints an allocation table (bytes/call, MITC4 vs MITC9) for the core assembly
# kernels — same functions covered by test/test_allocations.jl, run standalone
# here for reporting/presentation purposes.
#
#   julia --project=. AllocationTable.jl

using FerriteShells, LinearAlgebra
import FerriteShells: covariant_basis, director_field

function alloc_of(f)
    f(); f()
    @allocated f()
end

function loop_director(scv, u_e, n_nodes, n_qp)
    s = zero(Vec{3,Float64})
    for qp in 1:n_qp
        d, d₁, d₂ = director_field(scv, qp, u_e, n_nodes)
        s += d + d₁ + d₂
    end
    s
end
function loop_covariant(scv, u_e, n_nodes, n_qp)
    s = zero(Vec{3,Float64})
    for qp in 1:n_qp
        a₁, a₂ = covariant_basis(scv, qp, u_e, n_nodes)
        s += a₁ + a₂
    end
    s
end

elements = ((Triangle, RefTriangle, 1, MITC3, "MITC3"),
            (Quadrilateral, RefQuadrilateral, 1, MITC4, "MITC4"),
            (QuadraticTriangle, RefTriangle, 2, MITC6a, "MITC6a"),
            (QuadraticQuadrilateral, RefQuadrilateral, 2, MITC9, "MITC9"))

results = Dict{String,Vector{Tuple{String,Int}}}()

for elem in elements
    (Q,R,O,M,label) = elem
    grid     = shell_grid(generate_grid(Q, (2, 2)))
    addnodeset!(grid, "all", x->true)
    ip       = Lagrange{R, O}()
    qr       = QuadratureRule{R}(O+1)
    fqr      = FacetQuadratureRule{R}(O+1)
    scv_mitc = ShellCellValues(qr, ip, ip; mitc=M)
    scv      = ShellCellValues(qr, ip, ip)
    dh = DofHandler(grid)
    add!(dh, :u, ip^3); add!(dh, :θ, ip^2); close!(dh)
    mat  = LinearElastic(1.0e6, 0.3, 0.1)
    reinit!(scv_mitc, first(CellIterator(dh)))
    n_e     = ndofs_per_cell(dh)
    n_nodes = getnbasefunctions(scv_mitc.ip_shape)
    n_qp    = getnquadpoints(scv_mitc)
    u_e     = 0.001 .* randn(n_e)
    ke      = zeros(n_e, n_e)
    re      = zeros(n_e)
    f_ext   = zeros(ndofs(dh))

    rows = Tuple{String,Int}[]
    push!(rows, ("director_field (loop)", alloc_of(() -> loop_director(scv_mitc, u_e, n_nodes, n_qp))))
    push!(rows, ("covariant_basis (loop)", alloc_of(() -> loop_covariant(scv_mitc, u_e, n_nodes, n_qp))))
    push!(rows, ("assemble_pressure!", alloc_of(() -> assemble_pressure!(re, scv_mitc, u_e, 1.0))))
    push!(rows, ("assemble_pressure_tangent!", alloc_of(() -> assemble_pressure_tangent!(ke, scv_mitc, u_e, 1.0))))
    push!(rows, ("assemble_traction!", alloc_of(() -> assemble_traction!(f_ext, dh, getfacetset(grid, "left"), ip, fqr, Vec{3}((0.,0.,1.))))))
    push!(rows, ("apply_pointload!", alloc_of(() -> apply_pointload!(f_ext, dh, "all", Vec{3}((0.,0.,1.))))))
    push!(rows, ("mass_matrix!", alloc_of(() -> mass_matrix!(ke, scv_mitc, 1.0, mat))))
    push!(rows, ("membrane_residuals_RM!", alloc_of(() -> membrane_residuals_RM!(re, scv_mitc, u_e, mat))))
    push!(rows, ("bending_residuals_RM!", alloc_of(() -> bending_residuals_RM!(re, scv_mitc, u_e, mat))))
    push!(rows, ("membrane_tangent_RM!", alloc_of(() -> membrane_tangent_RM!(ke, scv_mitc, u_e, mat))))
    push!(rows, ("bending_tangent_RM!", alloc_of(() -> bending_tangent_RM!(ke, scv_mitc, u_e, mat))))
    push!(rows, ("bending_tangent_RM! (no MITC)", alloc_of(() -> bending_tangent_RM!(ke, scv, u_e, mat))))
    full() = (fill!(ke, 0.0); fill!(re, 0.0);
              membrane_residuals_RM!(re, scv_mitc, u_e, mat); bending_residuals_RM!(re, scv_mitc, u_e, mat);
              membrane_tangent_RM!(ke, scv_mitc, u_e, mat);   bending_tangent_RM!(ke, scv_mitc, u_e, mat))
    push!(rows, ("full per-cell sweep", alloc_of(full)))

    results[label] = rows
end

println()
println("| Function | MITC3 (bytes) | MITC4 (bytes) | MITC6a (bytes) | MITC9 (bytes) |")
println("|---|---|---|---|---|")
mitc3 = Dict(results["MITC3"])
mitc4 = Dict(results["MITC4"])
mitc6a = Dict(results["MITC6a"])
mitc9 = Dict(results["MITC9"])
for (name, _) in results["MITC3"]
    println("| $name | $(mitc3[name]) | $(mitc4[name]) | $(mitc6a[name]) | $(mitc9[name]) |")
end
