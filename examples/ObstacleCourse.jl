using FerriteShells, CairoMakie, FileIO

const IMG_DIR = joinpath(@__DIR__, "..", "docs", "src", "images")

configs = [
    (Triangle,               1, RefTriangle,      "Lagrange{RefTriangle, 1}"),
    (Quadrilateral,          1, RefQuadrilateral, "Lagrange{RefQuadrilateral, 1}"),
    (QuadraticTriangle,      2, RefTriangle,      "Lagrange{RefTriangle, 2}"),
    (QuadraticQuadrilateral, 2, RefQuadrilateral, "Lagrange{RefQuadrilateral, 2}"),
]

# RM/MITC shear treatment: MITC4/MITC9 for the quadrilateral family, MITC3/MITC6a for
# the triangle family (Lee & Bathe assumed-shear tying), selected by (element, order).
mitc_for(::Type{RefQuadrilateral}, order) = order == 1 ? MITC4 : MITC9
mitc_for(::Type{RefTriangle},      order) = order == 1 ? MITC3 : MITC6a

function cooks_membrane()
    function create_cook_grid(nx, ny; primitive=Quadrilateral)
        corners = [Ferrite.Vec{2}(( 0.0,  0.0)), Ferrite.Vec{2}((48.0, 44.0)),
                Ferrite.Vec{2}((48.0, 60.0)), Ferrite.Vec{2}(( 0.0, 44.0))]
        return generate_grid(primitive, (nx, ny), corners) |> shell_grid # embed in into a 3D space
    end

    function assemble_membrane!(K, r, dh, scv, u, mat)
        n = ndofs_per_cell(dh)
        ke = zeros(n, n)
        re  = zeros(n)
        assembler = start_assemble(K, r)
        for cell in CellIterator(dh)
            fill!(ke, 0.0); fill!(re, 0.0)
            reinit!(scv, cell) # prepares reference geometry
            u_e = u[celldofs(cell)]
            membrane_tangent_KL!(ke, scv, u_e, mat)
            membrane_residuals_KL!(re, scv, u_e, mat)
            assemble!(assembler, celldofs(cell), ke, re)
        end
    end

    function cooks_membrane_solve(n; primitive=Quadrilateral, order=1, element=RefQuadrilateral)
        # number of cells
        grid = create_cook_grid(2n, n; primitive=primitive)

        # facesets for boundary conditions
        addfacetset!(grid, "clamped", x -> norm(x[1]) ≈ 0.0)
        addfacetset!(grid, "traction", x -> norm(x[1]) ≈ 48.0)
        addnodeset!(grid, "nodes", x -> true)

        # interpolation order
        ip = Lagrange{element, order}()
        qr = QuadratureRule{element}(order+1)

        # cell (shell) values
        scv = ShellCellValues(qr, ip, ip)
        fqr = FacetQuadratureRule{element}(order+1)

        # degrees of freedom for displacements (pure membrane test)
        dh = DofHandler(grid)
        add!(dh, :u, ip^3)
        close!(dh)

        # material model
        mat = LinearElastic(1.0, 1/3)

        # boundary conditions
        dbc = ConstraintHandler(dh)
        add!(dbc, Dirichlet(:u, getfacetset(dh.grid, "clamped"), x -> zero(x), [1,2,3]))
        add!(dbc, Dirichlet(:u, getnodeset(dh.grid, "nodes"),    x -> [0.0], [3]))
        close!(dbc)

        # stiffness matrix and residuals vector construction and assembly
        Ke = allocate_matrix(dh)
        f = zeros(ndofs(dh))
        assemble_membrane!(Ke, f, dh, scv, zeros(ndofs(dh)), mat)

        # traction force assembly, force of 1N on the face, split into 16 units (length of face)
        assemble_traction!(f, dh, getfacetset(grid, "traction"), ip, fqr, (0.0, 1.0/16, 0.0))

        # apply BCs and solve (\) figures out the best linear solver to use
        apply!(Ke, f, dbc)
        ue = Ke \ f
        # extract solution at point
        ph     = PointEvalHandler(grid, [Ferrite.Vec{3}((48.0, 52.0, 0.0))])
        u_eval = first(evaluate_at_points(ph, dh, ue, :u))
        return u_eval[2]
    end

    # resolution sweep
    N = [2,4,8,16,32]
    fig = Figure(size=(800, 400))
    ax0 = Axis(fig[1, 1], aspect = DataAspect(), title="Deformed mesh Lagrange{RefQuadrilateral, 2}")
    ax1 = Axis(fig[1, 2], xlabel="Number of elements", ylabel="vertical tip displacement u₂",
               title="Convergence of vertical tip displacement")
    hlines!(ax1, 23.95, 0, 32, color=:black, linestyle=:dash, label="Reference", linewidth=2)
    for (prim, order, elem, label) in configs
        res = [cooks_membrane_solve(n; primitive=prim, order=order, element=elem) for n in N]
        lines!(ax1, N, res, label=label, linewidth=2)
    end
    img = load(joinpath(IMG_DIR, "cooks_membrane.png"))
    image!(ax0, rotr90(img))
    axislegend(ax1, position=:rb)
    hidespines!(ax0)
    hidedecorations!(ax0)
    xlims!(ax1, 0, maximum(N))
    ylims!(ax1, 0, 30)
    save(joinpath(IMG_DIR, "cooks_membrane_convergence.png"), fig)
    fig
end

function scordelis_lo_roof()
    # Scordelis-Lo roof — Reissner-Mindlin shell (1/4)
    function scordelis_lo_grid(ns; primitive=Quadrilateral)
        R_sl, L_sl, Φ_sl = 25.0, 50.0, 40π/180
        g = shell_grid(generate_grid(primitive, (ns, ns), Ferrite.Vec{2}((-Φ_sl, 0.0)), Ferrite.Vec{2}((Φ_sl, L_sl)));
                       map = n -> (n.x[2], R_sl * cos(n.x[1]), R_sl * sin(n.x[1])))
        addnodeset!(g, "diaphragm", x -> x[1] ≈ 0.0 || x[1] ≈ L_sl)
        addnodeset!(g, "ref_point", x -> abs(x[1] - L_sl/2) < 1e-8 && abs(x[2] - R_sl*cos(Φ_sl)) < 1e-8 &&
                                         abs(x[3] - R_sl*sin(Φ_sl)) < 1e-8)
        return g
    end

    function scordelis_lo_solve(ns; primitive=Quadrilateral, order=1, element=RefQuadrilateral)
        ip  = Lagrange{element, order}()
        qr  = QuadratureRule{element}(order + 1)
        scv = ShellCellValues(qr, ip, ip; mitc=mitc_for(element, order))
        mat = LinearElastic(4.32e8, 0.0, 0.25)

        grid = scordelis_lo_grid(ns; primitive=primitive)
        dh   = DofHandler(grid)
        add!(dh, :u, ip^3); add!(dh, :θ, ip^2); close!(dh)
        n_base = getnbasefunctions(ip)

        K  = allocate_matrix(dh)
        f  = zeros(ndofs(dh))
        asmb = start_assemble(K, zeros(ndofs(dh)))
        ke = zeros(5n_base, 5n_base); re = zeros(5n_base); fe = zeros(5n_base)
        q_sl = Ferrite.Vec{3}((0.0, -90.0, 0.0))
        for cell in CellIterator(dh)
            fill!(ke, 0.0); fill!(re, 0.0); fill!(fe, 0.0)
            reinit!(scv, cell)
            u0 = zeros(5n_base)
            membrane_tangent_RM!(ke, scv, u0, mat)
            bending_tangent_RM!(ke, scv, u0, mat)
            sd = shelldofs(cell)
            assemble!(asmb, sd, ke, re)
            for qp in 1:getnquadpoints(scv)
                ξ  = scv.qr.points[qp]; dΩ = scv.detJdV[qp]
                for I in 1:n_base
                    NI = Ferrite.reference_shape_value(ip, ξ, I)
                    @views fe[5I-4:5I-2] .+= NI * q_sl * dΩ
                end
            end
            @views f[sd] .+= fe
        end

        dbc = ConstraintHandler(dh)
        add!(dbc, Dirichlet(:u, getnodeset(grid, "diaphragm"), x -> zeros(2), [2, 3]))
        close!(dbc); Ferrite.update!(dbc, 0.0); apply!(K, f, dbc)

        u_sol = K \ f

        ref_nodes = collect(getnodeset(grid, "ref_point"))
        @assert length(ref_nodes) == 1
        for cell in CellIterator(dh)
            for (I, gid) in enumerate(getnodes(cell))
                if gid == ref_nodes[1]
                    cd = celldofs(cell)
                    return -u_sol[cd[3I-1]]  # y-component of :u
                end
            end
        end
        error("ref_point node not found in any cell")
    end

    # resolution sweep
    N = [8,16,32,64]
    fig = Figure(size=(800, 400))
    ax0 = Axis(fig[1, 1], aspect = DataAspect(), title="Deformed mesh Lagrange{RefQuadrilateral, 2}")
    ax1 = Axis(fig[1, 2], xlabel="Number of elements", ylabel="vertical displacement u₂",
               title="Convergence of vertical tip displacement")
    hlines!(ax1, 0.3024, 0, 32, color=:black, linestyle=:dash, label="Reference", linewidth=2)
    for (prim, order, elem, label) in configs
        res = [scordelis_lo_solve(n; primitive=prim, order=order, element=elem) for n in N]
        lines!(ax1, N, res, label=label, linewidth=2)
    end
    img = load(joinpath(IMG_DIR, "scoreldis_lo_roof.png"))
    image!(ax0, rotr90(img))
    axislegend(ax1, position=:rb)
    hidespines!(ax0)
    hidedecorations!(ax0)
    xlims!(ax1, 0, maximum(N))
    ylims!(ax1, 0, 0.35)
    save(joinpath(IMG_DIR, "scordelis_lo_roof_convergence.png"), fig)
    fig
end

function pinched_cylinder()
    # Pinched cylinder — Reissner-Mindlin shell (1/8 symmetry model)
    function pinched_cylinder_grid(ns, na; primitive=Quadrilateral)
        g = shell_grid(generate_grid(primitive, (ns, na), Ferrite.Vec{2}((0.0, 0.0)), Ferrite.Vec{2}((π/2, 600.0/2)));
                    map = n -> (n.x[2], 300.0 * sin(n.x[1]), 300.0 * cos(n.x[1])))
        addnodeset!(g, "diaphragm", x -> x[1] ≈ 0.0)
        addnodeset!(g, "sym_axial", x -> x[1] ≈ 600.0/2)
        addnodeset!(g, "sym_theta0", x -> abs(x[2]) < 1e-6)
        addnodeset!(g, "sym_theta90", x -> abs(x[3]) < 1e-6)
        addnodeset!(g, "load_point", x -> x[1] ≈ 600.0/2 && abs(x[2]) < 1e-6 && abs(x[3] - 300.0) < 1e-6)
        return g
    end

    function solver_pinched_cylinder(n; primitive=Quadrilateral, order=1, element=RefQuadrilateral)
        # interplation space
        ip  = Lagrange{element, order}()
        qr  = QuadratureRule{element}(order + 1)
        scv = ShellCellValues(qr, ip, ip; mitc=mitc_for(element, order))

        # material
        mat = LinearElastic(3.0e6, 0.3, 3.0)

        # make grid
        grid = pinched_cylinder_grid(n, n; primitive=primitive)
        nf   = NodeFrames(grid, ip)  # per-node averaged frames (curved-shell frame consistency)

        # degrees of freedom
        dh   = DofHandler(grid)
        add!(dh, :u, ip^3)
        add!(dh, :θ, ip^2)
        close!(dh)

        # assembly
        n_base = getnbasefunctions(ip)
        K  = allocate_matrix(dh)
        f  = zeros(ndofs(dh))
        asmb = start_assemble(K, zeros(ndofs(dh)))
        ke = zeros(5n_base, 5n_base); re = zeros(5n_base)

        for cell in CellIterator(dh)
            fill!(ke, 0.0); fill!(re, 0.0)
            reinit!(scv, cell, nf)
            u0 = zeros(5n_base)
            membrane_tangent_RM!(ke, scv, u0, mat)
            bending_tangent_RM!(ke, scv, u0, mat)
            sd = shelldofs(cell)
            assemble!(asmb, sd, ke, re)
        end

        apply_pointload!(f, dh, "load_point", Ferrite.Vec{3}((0.0, 0.0, -1/4)))

        dbc = ConstraintHandler(dh)
        add!(dbc, Dirichlet(:u, getnodeset(grid, "diaphragm"),   x -> zeros(2), [2, 3]))
        add!(dbc, Dirichlet(:u, getnodeset(grid, "sym_axial"),   x -> 0.0, [1]))
        add!(dbc, Dirichlet(:u, getnodeset(grid, "sym_theta0"),  x -> 0.0, [2]))
        add!(dbc, Dirichlet(:u, getnodeset(grid, "sym_theta90"), x -> 0.0, [3]))
        add!(dbc, Dirichlet(:θ, getnodeset(grid, "sym_theta0"),  x -> 0.0, [2]))
        add!(dbc, Dirichlet(:θ, getnodeset(grid, "sym_theta90"), x -> 0.0, [2]))
        add!(dbc, Dirichlet(:θ, getnodeset(grid, "sym_axial"),   x -> 0.0, [1]))
        close!(dbc); Ferrite.update!(dbc, 0.0); apply!(K, f, dbc)

        u_sol = K \ f

        # extract solution at point
        ph     = PointEvalHandler(grid, [Ferrite.Vec{3}(([300.0, 0.0, 300.0]))])
        u_eval = first(evaluate_at_points(ph, dh, u_sol, :u))
        return -u_eval[3]
    end

    # resolution sweep
    N = [4,8,16,32]
    fig = Figure(size=(800, 400))
    ax0 = Axis(fig[1, 1], aspect = DataAspect(), title="Deformed mesh Lagrange{RefQuadrilateral, 2}")
    ax1 = Axis(fig[1, 2], xlabel="Number of elements", ylabel="vertical displacement u₂", title="Convergence of vertical tip displacement")
    hlines!(ax1, 1.8248e-5, 0, 32, color=:black, linestyle=:dash, label="Reference", linewidth=2)
    for (prim, order, elem, label) in configs
        res = [solver_pinched_cylinder(n; primitive=prim, order=order, element=elem) for n in N]
        lines!(ax1, N, res, label=label, linewidth=2)
    end
    img = load(joinpath(IMG_DIR, "pinched_cylinder.png"))
    image!(ax0, rotr90(img))
    axislegend(ax1, position=:rb)
    hidespines!(ax0)
    hidedecorations!(ax0)
    xlims!(ax1, 0, maximum(N))
    ylims!(ax1, 0, 2e-5)
    save(joinpath(IMG_DIR, "pinched_cylinder_convergence.png"), fig)
    fig
end

function pinched_hemisphere()
    # Pinched hemisphere — Reissner-Mindlin shell (1/8 symmetry model)
    #
    # NOTE: as of this writing, even master's validated PinchedHemisphere.jl example
    # (single Q9+MITC9 config, n=32) returns u_x(A) ≈ 3.7e-4 against the 0.0924
    # reference — off by ~250x, not just "slow convergence". There is no regression
    # test covering this benchmark. Treat results from this function as unverified
    # until that is separately investigated.
    function hemisphere_grid(n; primitive=Quadrilateral)
        R=10.0; θ_hole_deg=18.0; θ_min = θ_hole_deg * π / 180
        g = shell_grid(generate_grid(primitive, (n, n), Ferrite.Vec{2}((θ_min, 0.0)), Ferrite.Vec{2}((π/2, π/2)));
                       map = nd -> (R*sin(nd.x[1])*cos(nd.x[2]), R*sin(nd.x[1])*sin(nd.x[2]), R*cos(nd.x[1])))
        addfacetset!(g, "sym_phi0",  x -> abs(x[2]) < 1e-10)
        addfacetset!(g, "sym_phi90", x -> abs(x[1]) < 1e-10)
        addnodeset!(g, "sym_phi0_n",  x -> abs(x[2]) < 1e-9)
        addnodeset!(g, "sym_phi90_n", x -> abs(x[1]) < 1e-9)
        addnodeset!(g, "load_A", x -> abs(x[3]) < 1e-6 && abs(x[2]) < 1e-6 && x[1] > 0.5R)
        addnodeset!(g, "load_B", x -> abs(x[3]) < 1e-6 && abs(x[1]) < 1e-6 && x[2] > 0.5R)
        return g
    end

    function solve_pinched_hemisphere(n; primitive=Quadrilateral, order=1, element=RefQuadrilateral)
        # interplation space
        ip  = Lagrange{element, order}()
        qr  = QuadratureRule{element}(order + 1)
        scv = ShellCellValues(qr, ip, ip; mitc=mitc_for(element, order))

        # material
        mat = LinearElastic(6.825e7, 0.3, 0.04)

        # make grid amd a NodeFrame
        grid = hemisphere_grid(n; primitive=primitive)
        nf = NodeFrames(grid, ip)

        # degrees of freedom
        dh = DofHandler(grid)
        add!(dh, :u, ip^3)
        add!(dh, :θ, ip^2)
        close!(dh)

        # boundary conditions
        ch = ConstraintHandler(dh)
        add!(ch, Dirichlet(:u, getfacetset(grid, "sym_phi0"),  x -> 0.0, [2]))
        add!(ch, Dirichlet(:u, getfacetset(grid, "sym_phi90"), x -> 0.0, [1]))
        add_director_symmetry!(ch, dh, nf, "sym_phi0_n",  Ferrite.Vec{3}((0.0, 1.0, 0.0)))
        add_director_symmetry!(ch, dh, nf, "sym_phi90_n", Ferrite.Vec{3}((1.0, 0.0, 0.0)))
        close!(ch); Ferrite.update!(ch, 0.0)

        #  allocate matrices and vectors
        n_base = getnbasefunctions(ip)
        K      = allocate_matrix(dh, ch)   # ch: the affine constraints add coupling entries
        f      = zeros(ndofs(dh))
        ke     = zeros(5n_base, 5n_base)
        re     = zeros(5n_base)

        # assemble once
        asm = start_assemble(K, zeros(ndofs(dh)))
        for cell in CellIterator(dh)
            fill!(ke, 0.0)
            reinit!(scv, cell, nf)   # per-node frames — the frame the symmetry BC is written in
            u0 = zeros(5n_base)
            membrane_tangent_RM!(ke, scv, u0, mat)
            bending_tangent_RM!(ke, scv, u0, mat)
            assemble!(asm, shelldofs(cell), ke, re)
        end

        # apply loading
        apply_pointload!(f, dh, "load_A", Ferrite.Vec{3}((-1.0, 0.0, 0.0)))
        apply_pointload!(f, dh, "load_B", Ferrite.Vec{3}(( 0.0, 1.0, 0.0)))
        apply!(K, f, ch)

        #solve and time it
        u_sol = K \ f
        apply!(u_sol, ch)   # recover the affine-constrained φ DOFs

        # extract solution at point
        ph     = PointEvalHandler(grid, [grid.nodes[first(grid.nodesets["load_A"])].x])
        u_eval = first(evaluate_at_points(ph, dh, u_sol, :u))
        return -u_eval[1]
    end

    # resolution sweep
    N = [4,8,16,32]
    fig = Figure(size=(800, 400))
    ax0 = Axis(fig[1, 1], aspect = DataAspect(), title="Deformed mesh Lagrange{RefQuadrilateral, 2}")
    ax1 = Axis(fig[1, 2], xlabel="Number of elements", ylabel="horizontal displacement u₁ at A",
               title="Convergence of horizontal tip displacement")
    hlines!(ax1, 0.0924, 0, 32, color=:black, linestyle=:dash, label="Reference", linewidth=2)
    for (prim, order, elem, label) in configs
        res = [solve_pinched_hemisphere(n; primitive=prim, order=order, element=elem) for n in N]
        lines!(ax1, N, res, label=label, linewidth=2)
    end
    img = load(joinpath(IMG_DIR, "pinched_hemisphere.png"))  # no dedicated hemisphere schematic yet
    image!(ax0, rotr90(img))
    axislegend(ax1, position=:rb)
    hidespines!(ax0)
    hidedecorations!(ax0)
    xlims!(ax1, 0, maximum(N))
    ylims!(ax1, 0, 0.1)
    # save(joinpath(IMG_DIR, "pinched_hemisphere_convergence.png"), fig)
    fig
end

function hyperbolic_paraboloid()
    # Partly clamped hyperbolic paraboloid — Reissner-Mindlin shell
    # Lee & Bathe (2005), Comput. Struct. 83:69-90, §3.4.2 / Chapelle-Bathe locking benchmark.
    # Surface z = x²-y² on [-1/2,1/2]², clamped along one straight edge (here y=-1/2), free
    # elsewhere, self-weight load q=80/area. Reference u_z at the midpoint of the opposite free
    # edge is quoted in the literature as ≈ -9.3137e-5 and ≈ -9.3355e-5 (two independent fine-mesh
    # solutions, clamped edge x=∓1/2 there); by the surface's own 90°-rotation+z-flip symmetry the
    # clamped-edge choice doesn't change the value at the corresponding point.
    function hp_grid(ns; primitive=Quadrilateral)
        L_hp = 1.0
        g = shell_grid(generate_grid(primitive, (ns, ns), Ferrite.Vec{2}((-L_hp/2, -L_hp/2)), Ferrite.Vec{2}((L_hp/2, L_hp/2)));
                       map = n -> (n.x[1], n.x[2], n.x[1]^2 - n.x[2]^2))
        addfacetset!(g, "clamped", x -> x[2] ≈ -L_hp/2)
        return g
    end

    function solve_hyperbolic_paraboloid(n; primitive=Quadrilateral, order=1, element=RefQuadrilateral)
        ip  = Lagrange{element, order}()
        qr  = QuadratureRule{element}(order + 1)
        scv = ShellCellValues(qr, ip, ip; mitc=mitc_for(element, order))
        mat = LinearElastic(2e11, 0.3, 0.01)

        grid = hp_grid(n; primitive=primitive)
        nf   = NodeFrames(grid, ip)  # per-node averaged frames (curved-shell frame consistency)
        dh   = DofHandler(grid)
        add!(dh, :u, ip^3); add!(dh, :θ, ip^2); close!(dh)
        n_base = getnbasefunctions(ip)

        K  = allocate_matrix(dh)
        f  = zeros(ndofs(dh))
        asmb = start_assemble(K, zeros(ndofs(dh)))
        ke = zeros(5n_base, 5n_base); re = zeros(5n_base); fe = zeros(5n_base)
        q_hp = Ferrite.Vec{3}((0.0, 0.0, -80.0))
        for cell in CellIterator(dh)
            fill!(ke, 0.0); fill!(re, 0.0); fill!(fe, 0.0)
            reinit!(scv, cell, nf)
            u0 = zeros(5n_base)
            membrane_tangent_RM!(ke, scv, u0, mat)
            bending_tangent_RM!(ke, scv, u0, mat)
            sd = shelldofs(cell)
            assemble!(asmb, sd, ke, re)
            for qp in 1:getnquadpoints(scv)
                ξ  = scv.qr.points[qp]; dΩ = scv.detJdV[qp]
                for I in 1:n_base
                    NI = Ferrite.reference_shape_value(ip, ξ, I)
                    @views fe[5I-4:5I-2] .+= NI * q_hp * dΩ
                end
            end
            @views f[sd] .+= fe
        end

        dbc = ConstraintHandler(dh)
        add!(dbc, Dirichlet(:u, getfacetset(grid, "clamped"), x -> zero(x), [1,2,3]))
        add!(dbc, Dirichlet(:θ, getfacetset(grid, "clamped"), x -> zeros(2), [1,2]))
        close!(dbc); Ferrite.update!(dbc, 0.0); apply!(K, f, dbc)

        u_sol = K \ f

        # midpoint of the free edge opposite the clamped one
        ph     = PointEvalHandler(grid, [Ferrite.Vec{3}((0.0, 0.5, -0.25))])
        u_eval = first(evaluate_at_points(ph, dh, u_sol, :u))
        return u_eval[3]
    end

    # resolution sweep
    N = [8,16,32,64]
    fig = Figure(size=(800, 400))
    ax0 = Axis(fig[1, 1], aspect = DataAspect(), title="Deformed mesh Lagrange{RefQuadrilateral, 2}")
    ax1 = Axis(fig[1, 2], xlabel="Number of elements", ylabel="vertical displacement u₃ at free-edge midpoint",
               title="Convergence of vertical displacement")
    hlines!(ax1, -9.335e-5, 0, 64, color=:black, linestyle=:dash, label="Reference", linewidth=2)
    for (prim, order, elem, label) in configs
        res = [solve_hyperbolic_paraboloid(n; primitive=prim, order=order, element=elem) for n in N]
        lines!(ax1, N, res, label=label, linewidth=2)
    end
    img = load(joinpath(IMG_DIR, "hyperbolic_paraboloid.png"))
    image!(ax0, rotr90(img))
    axislegend(ax1, position=:rt)
    hidespines!(ax0)
    hidedecorations!(ax0)
    xlims!(ax1, 0, maximum(N))
    ylims!(ax1, -1.1e-4, 0)
    save(joinpath(IMG_DIR, "hyperbolic_paraboloid_convergence.png"), fig)
    fig
end

# cooks_membrane()
# scordelis_lo_roof()
# pinched_cylinder()
# pinched_hemisphere()
hyperbolic_paraboloid()
