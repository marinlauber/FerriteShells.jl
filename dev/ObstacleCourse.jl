using FerriteShells

configs = [
    (Triangle,               1, RefTriangle,      "Lagrange{RefTriangle, 1}"),
    (Quadrilateral,          1, RefQuadrilateral, "Lagrange{RefQuadrilateral, 1}"),
    (QuadraticTriangle,      2, RefTriangle,      "Lagrange{RefTriangle, 2}"),
    (QuadraticQuadrilateral, 2, RefQuadrilateral, "Lagrange{RefQuadrilateral, 2}"),
]

# let
#     function create_cook_grid(nx, ny; primitive=Quadrilateral)
#         corners = [Ferrite.Vec{2}(( 0.0,  0.0)), Ferrite.Vec{2}((48.0, 44.0)),
#                 Ferrite.Vec{2}((48.0, 60.0)), Ferrite.Vec{2}(( 0.0, 44.0))]
#         return generate_grid(primitive, (nx, ny), corners) |> shell_grid # embed in into a 3D space
#     end

#     function assemble_membrane!(K, r, dh, scv, u, mat)
#         n = ndofs_per_cell(dh)
#         ke = zeros(n, n)
#         re  = zeros(n)
#         assembler = start_assemble(K, r)
#         for cell in CellIterator(dh)
#             fill!(ke, 0.0); fill!(re, 0.0)
#             reinit!(scv, cell) # prepares reference geometry
#             u_e = u[celldofs(cell)]
#             membrane_tangent_KL!(ke, scv, u_e, mat)
#             membrane_residuals_KL!(re, scv, u_e, mat)
#             assemble!(assembler, celldofs(cell), ke, re)
#         end
#     end

#     function cooks_membrane_solve(n; primitive=Quadrilateral, order=1, element=RefQuadrilateral)
#         # number of cells
#         grid = create_cook_grid(2n, n; primitive=primitive)

#         # facesets for boundary conditions
#         addfacetset!(grid, "clamped", x -> norm(x[1]) ≈ 0.0)
#         addfacetset!(grid, "traction", x -> norm(x[1]) ≈ 48.0)
#         addnodeset!(grid, "nodes", x -> true)

#         # interpolation order
#         ip = Lagrange{element, order}()
#         qr = QuadratureRule{element}(order+1)

#         # cell (shell) values
#         scv = ShellCellValues(qr, ip, ip)
#         fqr = FacetQuadratureRule{element}(order+1)

#         # degrees of freedom for displacements (pure membrane test)
#         dh = DofHandler(grid)
#         add!(dh, :u, ip^3)
#         close!(dh)

#         # material model
#         mat = LinearElastic(1.0, 1/3)

#         # boundary conditions
#         dbc = ConstraintHandler(dh)
#         add!(dbc, Dirichlet(:u, getfacetset(dh.grid, "clamped"), x -> zero(x), [1,2,3]))
#         add!(dbc, Dirichlet(:u, getnodeset(dh.grid, "nodes"),    x -> [0.0], [3]))
#         close!(dbc)

#         # stiffness matrix and residuals vector construction and assembly
#         Ke = allocate_matrix(dh)
#         f = zeros(ndofs(dh))
#         assemble_membrane!(Ke, f, dh, scv, zeros(ndofs(dh)), mat)

#         # traction force assembly, force of 1N on the face, split into 16 units (length of face)
#         assemble_traction!(f, dh, getfacetset(grid, "traction"), ip, fqr, (0.0, 1.0/16, 0.0))

#         # apply BCs and solve (\) figures out the best linear solver to use
#         apply!(Ke, f, dbc)
#         ue = Ke \ f
#         # extract solution at point
#         ph     = PointEvalHandler(grid, [Ferrite.Vec{3}((48.0, 52.0, 0.0))])
#         u_eval = first(evaluate_at_points(ph, dh, ue, :u))
#         return u_eval[2]
#     end

#     using CairoMakie,FileIO
#     N = [2,4,8,16,32]
#     f = Figure(size=(800, 400))
#     ax0 = Axis(f[1, 1], aspect = DataAspect(), title="Deformed mesh Lagrange{RefQuadrilateral, 2}")
#     ax1 = Axis(f[1, 2], xlabel="Number of elements", ylabel="vertical tip displacement u₂", title="Convergence of vertical tip displacement")
#     hlines!(ax1, 23.95, 0, 32, color=:black, linestyle=:dash, label="Reference", linewidth=2)
#     for (prim, order, elem, label) in configs
#         res = [cooks_membrane_solve(n; primitive=prim, order=order, element=elem) for n in N]
#         lines!(ax1, N, res, label=label, linewidth=2)
#     end
#     img = load("/home/marin/Workspace/FerriteShells.jl/docs/src/images/cooks_membrane.png")
#     image!(ax0, rotr90(img))
#     axislegend(ax1, position=:rb)
#     hidespines!(ax0)
#     hidedecorations!(ax0)
#     xlims!(ax1, 0, maximum(N))
#     ylims!(ax1, 0, 30)
#     # save(joinpath("/home/marin/Workspace/FerriteShells.jl/docs/src/images", "cooks_membrane_convergence.png"), f)
#     f
# end


# let
#     function scordelis_lo_grid(ns; primitive=Quadrilateral)
#         R_sl, L_sl, Φ_sl = 25.0, 50.0, 40π/180
#         g = shell_grid(generate_grid(primitive, (ns, ns), Ferrite.Vec{2}((-Φ_sl, 0.0)), Ferrite.Vec{2}((Φ_sl, L_sl)));
#                        map = n -> (n.x[2], R_sl * cos(n.x[1]), R_sl * sin(n.x[1])))
#         addnodeset!(g, "diaphragm", x -> x[1] ≈ 0.0 || x[1] ≈ L_sl)
#         addnodeset!(g, "ref_point", x -> abs(x[1] - L_sl/2) < 1e-8 && abs(x[2] - R_sl*cos(Φ_sl)) < 1e-8 &&
#                                          abs(x[3] - R_sl*sin(Φ_sl)) < 1e-8)
#         return g
#     end

#     function scordelis_lo_solve(ns; primitive=Quadrilateral, order=1, element=RefQuadrilateral)
#         ip  = Lagrange{element, order}()
#         qr  = QuadratureRule{element}(order + 1)
#         scv = ShellCellValues(qr, ip, ip)
#         mat = LinearElastic(4.32e8, 0.0, 0.25)

#         grid = scordelis_lo_grid(ns; primitive=primitive)
#         dh   = DofHandler(grid)
#         add!(dh, :u, ip^3); add!(dh, :θ, ip^2); close!(dh)
#         n_base = getnbasefunctions(ip)

#         K  = allocate_matrix(dh)
#         f  = zeros(ndofs(dh))
#         asmb = start_assemble(K, zeros(ndofs(dh)))
#         ke = zeros(5n_base, 5n_base); re = zeros(5n_base); fe = zeros(5n_base)
#         q_sl = Ferrite.Vec{3}((0.0, -90.0, 0.0))
#         for cell in CellIterator(dh)
#             fill!(ke, 0.0); fill!(re, 0.0); fill!(fe, 0.0)
#             reinit!(scv, cell)
#             u0 = zeros(5n_base)
#             membrane_tangent_RM_FD!(ke, scv, u0, mat)
#             bending_tangent_RM_FD!(ke, scv, u0, mat)
#             sd = shelldofs(cell)
#             assemble!(asmb, sd, ke, re)
#             for qp in 1:getnquadpoints(scv)
#                 ξ  = scv.qr.points[qp]; dΩ = scv.detJdV[qp]
#                 for I in 1:n_base
#                     NI = Ferrite.reference_shape_value(ip, ξ, I)
#                     @views fe[5I-4:5I-2] .+= NI * q_sl * dΩ
#                 end
#             end
#             @views f[sd] .+= fe
#         end

#         dbc = ConstraintHandler(dh)
#         add!(dbc, Dirichlet(:u, getnodeset(grid, "diaphragm"), x -> zeros(2), [2, 3]))
#         close!(dbc); Ferrite.update!(dbc, 0.0); apply!(K, f, dbc)

#         u_sol = K \ f

#         ref_nodes = collect(getnodeset(grid, "ref_point"))
#         @assert length(ref_nodes) == 1
#         for cell in CellIterator(dh)
#             for (I, gid) in enumerate(getnodes(cell))
#                 if gid == ref_nodes[1]
#                     cd = celldofs(cell)
#                     return -u_sol[cd[3I-1]]  # y-component of :u
#                 end
#             end
#         end
#         error("ref_point node not found in any cell")
#     end

#     using CairoMakie,FileIO
#     N = [8,16,32,64]
#     f = Figure(size=(800, 400))
#     ax0 = Axis(f[1, 1], aspect = DataAspect(), title="Deformed mesh Lagrange{RefQuadrilateral, 2}")
#     ax1 = Axis(f[1, 2], xlabel="Number of elements", ylabel="vertical displacement u₂", title="Convergence of vertical tip displacement")
#     hlines!(ax1, 0.3024, 0, 32, color=:black, linestyle=:dash, label="Reference", linewidth=2)
#     for (prim, order, elem, label) in configs
#         res = [scordelis_lo_solve(n; primitive=prim, order=order, element=elem) for n in N]
#         lines!(ax1, N, res, label=label, linewidth=2)
#     end
#     img = load("/home/marin/Workspace/FerriteShells.jl/docs/src/images/scoreldis_lo_roof.png")
#     image!(ax0, rotr90(img))
#     axislegend(ax1, position=:rb)
#     hidespines!(ax0)
#     hidedecorations!(ax0)
#     xlims!(ax1, 0, maximum(N))
#     ylims!(ax1, 0, 0.35)
#     # save(joinpath("/home/marin/Workspace/FerriteShells.jl/docs/src/images", "scordelis_lo_roof_convergence.png"), f)
#     f
# end

# let
#     # Pinched cylinder — Reissner-Mindlin shell (1/8 symmetry model)
#     function pinched_cylinder_grid(ns, na; primitive=Quadrilateral)
#         g = shell_grid(generate_grid(primitive, (ns, na), Ferrite.Vec{2}((0.0, 0.0)), Ferrite.Vec{2}((π/2, 600.0/2)));
#                     map = n -> (n.x[2], 300.0 * sin(n.x[1]), 300.0 * cos(n.x[1])))
#         addnodeset!(g, "diaphragm", x -> x[1] ≈ 0.0)
#         addnodeset!(g, "sym_axial", x -> x[1] ≈ 600.0/2)
#         addnodeset!(g, "sym_theta0", x -> abs(x[2]) < 1e-6)
#         addnodeset!(g, "sym_theta90", x -> abs(x[3]) < 1e-6)
#         addnodeset!(g, "load_point", x -> x[1] ≈ 600.0/2 && abs(x[2]) < 1e-6 && abs(x[3] - 300.0) < 1e-6)
#         return g
#     end

#     function solver_pinched_cylinder(n; primitive=Quadrilateral, order=1, element=RefQuadrilateral)
#         # interplation space
#         ip  = Lagrange{element, order}()
#         qr  = QuadratureRule{element}(order + 1)
#         scv = ShellCellValues(qr, ip, ip)

#         # material
#         mat = LinearElastic(3.0e6, 0.3, 3.0)

#         # make grid
#         grid = pinched_cylinder_grid(n, n; primitive=primitive)

#         # degrees of freedom
#         dh   = DofHandler(grid)
#         add!(dh, :u, ip^3)
#         add!(dh, :θ, ip^2)
#         close!(dh)

#         # assembly
#         n_base = getnbasefunctions(ip)
#         K  = allocate_matrix(dh)
#         f  = zeros(ndofs(dh))
#         asmb = start_assemble(K, zeros(ndofs(dh)))
#         ke = zeros(5n_base, 5n_base); re = zeros(5n_base)

#         for cell in CellIterator(dh)
#             fill!(ke, 0.0); fill!(re, 0.0)
#             reinit!(scv, cell)
#             u0 = zeros(5n_base)
#             membrane_tangent_RM_FD!(ke, scv, u0, mat)
#             bending_tangent_RM_FD!(ke, scv, u0, mat)
#             sd = shelldofs(cell)
#             assemble!(asmb, sd, ke, re)
#         end

#         apply_pointload!(f, dh, "load_point", Ferrite.Vec{3}((0.0, 0.0, -1/4)))

#         dbc = ConstraintHandler(dh)
#         add!(dbc, Dirichlet(:u, getnodeset(grid, "diaphragm"),   x -> zeros(2), [2, 3]))
#         add!(dbc, Dirichlet(:u, getnodeset(grid, "sym_axial"),   x -> 0.0, [1]))
#         add!(dbc, Dirichlet(:u, getnodeset(grid, "sym_theta0"),  x -> 0.0, [2]))
#         add!(dbc, Dirichlet(:u, getnodeset(grid, "sym_theta90"), x -> 0.0, [3]))
#         add!(dbc, Dirichlet(:θ, getnodeset(grid, "sym_theta0"),  x -> 0.0, [1]))
#         add!(dbc, Dirichlet(:θ, getnodeset(grid, "sym_theta90"), x -> 0.0, [1]))
#         add!(dbc, Dirichlet(:θ, getnodeset(grid, "sym_axial"),   x -> 0.0, [2]))
#         close!(dbc); Ferrite.update!(dbc, 0.0); apply!(K, f, dbc)

#         u_sol = K \ f

#         # extract solution at point
#         ph     = PointEvalHandler(grid, [Ferrite.Vec{3}(([300.0, 0.0, 300.0]))])
#         u_eval = first(evaluate_at_points(ph, dh, u_sol, :u))
#         return -u_eval[3]
#     end

#     using CairoMakie,FileIO
#     N = [4,8,16,32]
#     f = Figure(size=(800, 400))
#     ax0 = Axis(f[1, 1], aspect = DataAspect(), title="Deformed mesh Lagrange{RefQuadrilateral, 2}")
#     ax1 = Axis(f[1, 2], xlabel="Number of elements", ylabel="vertical displacement u₂", title="Convergence of vertical tip displacement")
#     hlines!(ax1, 1.8248e-5, 0, 32, color=:black, linestyle=:dash, label="Reference", linewidth=2)
#     for (prim, order, elem, label) in configs
#         res = [solver_pinched_cylinder(n; primitive=prim, order=order, element=elem) for n in N]
#         lines!(ax1, N, res, label=label, linewidth=2)
#     end
#     img = load("/home/marin/Workspace/FerriteShells.jl/docs/src/images/pinched_cylinder.png")
#     image!(ax0, rotr90(img))
#     axislegend(ax1, position=:rb)
#     hidespines!(ax0)
#     hidedecorations!(ax0)
#     xlims!(ax1, 0, maximum(N))
#     ylims!(ax1, 0, 2e-5)
#     # save(joinpath("/home/marin/Workspace/FerriteShells.jl/docs/src/images", "pinched_cylinder_convergence.png"), f)
#     f
# end

let
    # Pinched hemisphere — Reissner-Mindlin shell (1/8 symmetry model)
    function hemisphere_grid(n; primitive=Quadrilateral)
        R=10.0; θ_hole_deg=18.0; θ_min = θ_hole_deg * π / 180
        g = shell_grid(generate_grid(primitive, (n, n), Ferrite.Vec{2}((θ_min, 0.0)), Ferrite.Vec{2}((π/2, π/2)));
                       map = nd -> (R*sin(nd.x[1])*cos(nd.x[2]), R*sin(nd.x[1])*sin(nd.x[2]), R*cos(nd.x[1])))
        addfacetset!(g, "sym_phi0",  x -> abs(x[2]) < 1e-10)
        addfacetset!(g, "sym_phi90", x -> abs(x[1]) < 1e-10)
        addnodeset!(g, "load_A", x -> abs(x[3]) < 1e-6 && abs(x[2]) < 1e-6 && x[1] > 0.5R)
        addnodeset!(g, "load_B", x -> abs(x[3]) < 1e-6 && abs(x[1]) < 1e-6 && x[2] > 0.5R)
        return g
    end

    function solve_pinched_hemisphere(n; primitive=Quadrilateral, order=1, element=RefQuadrilateral)
        # interplation space
        ip  = Lagrange{element, order}()
        qr  = QuadratureRule{element}(order + 1)
        scv = ShellCellValues(qr, ip, ip)

        # material
        mat = LinearElastic(6.825e7, 0.3, 0.04)

        # make grid
        grid = hemisphere_grid(n; primitive=primitive)

        # degrees of freedom
        dh = DofHandler(grid)
        add!(dh, :u, ip^3)
        add!(dh, :θ, ip^2)
        close!(dh)

        # boundary conditions
        ch = ConstraintHandler(dh)
        add!(ch, Dirichlet(:u, getfacetset(grid, "sym_phi0"),  x -> 0.0, [2]))
        add!(ch, Dirichlet(:u, getfacetset(grid, "sym_phi90"), x -> 0.0, [1]))
        add!(ch, Dirichlet(:θ, getfacetset(grid, "sym_phi0"),  x -> 0.0, [2]))
        add!(ch, Dirichlet(:θ, getfacetset(grid, "sym_phi90"), x -> 0.0, [2]))
        close!(ch); Ferrite.update!(ch, 0.0)

        N      = ndofs(dh)
        n_base = getnbasefunctions(ip)
        K      = allocate_matrix(dh)
        f      = zeros(N)
        ke     = zeros(5n_base, 5n_base)
        re     = zeros(5n_base)

        asm = start_assemble(K, zeros(N))
        for cell in CellIterator(dh)
            fill!(ke, 0.0)
            reinit!(scv, cell)
            u0 = zeros(5n_base)
            membrane_tangent_RM_FD!(ke, scv, u0, mat)
            bending_tangent_RM_FD!(ke, scv, u0, mat)
            assemble!(asm, shelldofs(cell), ke, re)
        end

        apply_pointload!(f, dh, "load_A", Ferrite.Vec{3}((-1.0, 0.0, 0.0)))
        apply_pointload!(f, dh, "load_B", Ferrite.Vec{3}(( 0.0, 1.0, 0.0)))
        apply!(K, f, ch)
        u_sol = K \ f

        A_node = only(getnodeset(grid, "load_A"))
        u_x_A  = 0.0
        for cell in CellIterator(dh), (I, gid) in enumerate(getnodes(cell))
            gid == A_node || continue
            u_x_A = u_sol[celldofs(cell)[3I-2]]
        end
        return -u_x_A
    end

    using CairoMakie,FileIO
    N = [4,8,16,32]
    f = Figure(size=(800, 400))
    ax0 = Axis(f[1, 1], aspect = DataAspect(), title="Deformed mesh Lagrange{RefQuadrilateral, 2}")
    ax1 = Axis(f[1, 2], xlabel="Number of elements", ylabel="horizontal displacement u₁ at A", title="Convergence of horizontal tip displacement")
    hlines!(ax1, 0.0924, 0, 32, color=:black, linestyle=:dash, label="Reference", linewidth=2)
    for (prim, order, elem, label) in configs
        res = [solve_pinched_hemisphere(n; primitive=prim, order=order, element=elem) for n in N]
        @show res, N
        # lines!(ax1, N, res, label=label, linewidth=2)
    end
    img = load("/home/marin/Workspace/FerriteShells.jl/docs/src/images/pinched_hemisphere.png")
    image!(ax0, rotr90(img))
    axislegend(ax1, position=:rb)
    hidespines!(ax0)
    hidedecorations!(ax0)
    xlims!(ax1, 0, maximum(N))
    ylims!(ax1, 0, 0.1)
    # save(joinpath("/home/marin/Workspace/FerriteShells.jl/docs/src/images", "pinched_hemisphere_convergence.png"), f)
    f
end