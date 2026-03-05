using FerriteShells

function create_cook_grid(nx, ny)
    corners = [Vec{2}((0.0,  0.0)), Vec{2}((48.0, 44.0)),
               Vec{2}((48.0, 60.0)), Vec{2}((0.0,  44.0))]
    return generate_grid(Triangle, (nx, ny), corners) |> shell_grid
end

# integrate edge traction force into f
# DOF ordering assumed: :u field first (3 DOFs per node, interleaved u,v,w),
# :θ field second.  This matches Ferrite's ordering when fields are added in that order.
function assemble_traction_force!(f, dh, facetset, traction)
    edge_local_nodes = Ferrite.reference_facets(RefTriangle)  # ((1,2),(2,3),(3,1))
    n_dpc = ndofs_per_cell(dh)
    fe    = zeros(n_dpc)
    for fc in FacetIterator(dh, facetset)
        x  = getcoordinates(fc)
        fn = fc.current_facet_id             # local facet index: 1, 2, or 3
        ia, ib = edge_local_nodes[fn]        # local node indices on this edge
        edge_len = norm(x[ib] - x[ia])
        fill!(fe, 0.0)
        # 1-point midpoint quadrature: both edge nodes receive equal weight 0.5
        # (exact for the linear shape functions used here)
        for (node, N) in ((ia, 0.5), (ib, 0.5))
            for c in 1:3    # u, v, w components
                fe[3(node-1)+c] += N * traction[c] * edge_len
            end
        end
        f[celldofs(fc)] .+= fe
    end
    return nothing
end


# make the problem
n= 16
grid = create_cook_grid(n, n)
addfacetset!(grid, "clamped",  x -> x[1] ≈ 0.0)
addfacetset!(grid, "traction", x -> x[1] ≈ 48.0)

# interpolation
ip = Lagrange{RefTriangle,1}()
# quadrature for membrane
qr = QuadratureRule{RefTriangle}(1)

# membrane/bending and shear ShellCellValues
scv = ShellCellValues(qr, ip, ip)

# set degrees of freedom
dh = DofHandler(grid)
add!(dh, :u, ip^3)   # translational DOFs (u, v, w)
add!(dh, :d, ip^3)   # director DOFs (d₁, d₂, d₃)
close!(dh)

# apply boundary conditions
dbc = ConstraintHandler(dh)
add!(dbc, Dirichlet(:u, getfacetset(dh.grid, "clamped"), x -> zero(x),   [1, 2, 3]))
add!(dbc, Dirichlet(:d, getfacetset(dh.grid, "clamped"), x -> [0.0, 0.0], [1, 2]))
close!(dbc)

# 40 Y. Ko et al. / Computers and Structures 192 (2017) 34–49 http://dx.doi.org/10.1016/j.compstruc.2017.07.003
E = 1.0        # stiffness (N)
t = 0.5        # thickness (dm)
ν = 1.0/3.0
mat = LinearMembraneMaterial(E, ν, t)

# assemble element stiffness matrices into K
function assemble_shell!(K, dh)
    assembler = start_assemble(K)
    for cell in CellIterator(dh)
        reinit!(scv, cell) # prepares reference geometry
        for qp in 1:getnquadpoints(scv)
            geom = update!(scv, cell) # computes current geometry for that quadrature
            a1 = geom.a1
            a2 = geom.a2
            A_metric = geom.A_metric
            a_metric = geom.a_metric
            E = 0.5 * (a_metric - A_metric)
            N = membrane_stress(mat, E)
            w = getweight(scv, qp) * geom.detJ
            # compute element stiffness contribution and assemble into K
            # ...
        end
        # assemble!(assembler, celldofs(cell), )
    end
    return nothing
end


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