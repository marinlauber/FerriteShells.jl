using FerriteShells

function create_cook_grid(nx, ny; primitive=Quadrilateral)
    corners = [Ferrite.Vec{2}(( 0.0,  0.0)), Ferrite.Vec{2}((48.0, 44.0)),
               Ferrite.Vec{2}((48.0, 60.0)), Ferrite.Vec{2}(( 0.0, 44.0))]
    generate_grid(primitive, (nx, ny), corners) |> shell_grid
end

function assemble_membrane!(K, r, dh, scv, u, mat)
    n = ndofs_per_cell(dh)
    ke = zeros(n, n)
    re = zeros(n)
    assembler = start_assemble(K, r)
    for cell in CellIterator(dh)
        fill!(ke, 0.0); fill!(re, 0.0)
        reinit!(scv, cell)
        u_e = u[celldofs(cell)]
        membrane_tangent_KL!(ke, scv, u_e, mat)
        membrane_residuals_KL!(re, scv, u_e, mat)
        assemble!(assembler, celldofs(cell), ke, re)
    end
end

function main(n; primitive=Quadrilateral, order=1, element=RefQuadrilateral)
    grid = create_cook_grid(2n, n; primitive=primitive)

    addfacetset!(grid, "clamped",  x -> norm(x[1]) ≈ 0.0)
    addfacetset!(grid, "traction", x -> norm(x[1]) ≈ 48.0)
    addnodeset!(grid,  "nodes",    x -> true)

    ip  = Lagrange{element, order}()
    qr  = QuadratureRule{element}(order+1)
    scv = ShellCellValues(qr, ip, ip)
    fqr = FacetQuadratureRule{element}(order+1)

    dh = DofHandler(grid)
    add!(dh, :u, ip^3)
    close!(dh)

    mat = LinearElastic(1.0, 1/3)

    dbc = ConstraintHandler(dh)
    add!(dbc, Dirichlet(:u, getfacetset(dh.grid, "clamped"), x -> zero(x), [1,2,3]))
    add!(dbc, Dirichlet(:u, getnodeset(dh.grid, "nodes"),    x -> [0.0],   [3]))
    close!(dbc)

    Ke = allocate_matrix(dh)
    r  = zeros(ndofs(dh))
    assemble_membrane!(Ke, r, dh, scv, zeros(ndofs(dh)), mat)

    # Save external traction before apply! modifies f.
    # Π = ½ f_ext · u is the correct linearized strain energy (K u = f_ext, so u^T K u = f_ext · u).
    f_ext = zeros(ndofs(dh))
    assemble_traction!(f_ext, dh, getfacetset(grid, "traction"), ip, fqr, (0.0, 1.0/16, 0.0))

    f = copy(f_ext)
    apply!(Ke, f, dbc)
    ue = Ke \ f
    # midpoint of free edge
    ph     = PointEvalHandler(grid, [Ferrite.Vec{3}((48.0, 52.0, 0.0))])
    u_eval = first(evaluate_at_points(ph, dh, ue, :u))
    Π = 0.5 * dot(f_ext, ue)
    u_eval[2], Π
end

using CairoMakie, FileIO

const N = [2, 4, 8, 16, 32, 64]

# Reference strain energy from a fine Q9 mesh (Π_ref ≈ Π_exact)
_, Π_ref = main(128; primitive=QuadraticQuadrilateral, order=2, element=RefQuadrilateral)

configs = [
    (Triangle,               1, RefTriangle,      "T3"),
    (Quadrilateral,          1, RefQuadrilateral, "Q4"),
    (QuadraticTriangle,      2, RefTriangle,      "T6"),
    (QuadraticQuadrilateral, 2, RefQuadrilateral, "Q9"),
]

fig = Figure(size=(800, 400))
ax1 = Axis(fig[1, 1], xlabel="n", ylabel="Vertical tip displacement u₂",
           title="Cook's membrane — tip displacement")
ax2 = Axis(fig[1, 2], xlabel="h ∝ 1/n", ylabel="S-norm error ‖e‖_S",
           title="Cook's membrane — S-norm convergence", xscale=log10, yscale=log10)

lines!(ax1, N, fill(23.95, length(N)), color=:black, linestyle=:dash, label="Reference")
for (prim, order, elem, label) in configs
    results = [main(n; primitive=prim, order=order, element=elem) for n in N]
    tips    = getindex.(results, 1)
    s_errs  = sqrt.(abs.(2 .* (Π_ref .- getindex.(results, 2))))
    lines!(ax1, N, tips, label=label, linewidth=2)
    scatterlines!(ax2, 1.0 ./ N, s_errs, linewidth=2)
end

# Reference convergence slopes
h = 1.0 ./ [N[1], N[end]]
lines!(ax2, h, 0.8  .* h,    linestyle=:dash, color=:black, label="O(h)")
lines!(ax2, h, 4.0  .* h.^2, linestyle=:dot,  color=:black, label="O(h²)")

axislegend(ax1, position=:rb)
axislegend(ax2, position=:rb)

save(joinpath("/home/marin/Workspace/FerriteShells.jl/docs/src/images", "cooks_membrane_snorm.png"), fig)
fig
