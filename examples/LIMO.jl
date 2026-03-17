using FerriteShells

# colors the surface in the mesh by their ID from the *.inp file
function color(vtk, grid)
    cellsets = keys(Ferrite.getcellsets(grid))
    z = zeros(Ferrite.getncells(grid))
    for cellset in cellsets
        z[collect(Ferrite.getcellset(grid, cellset))] .= parse(Float64, split(cellset, "_")[end])
    end
    write_cell_data(vtk, z, "SRF")
    return vtk
end

using OrderedCollections
function addcustomfacetset!(grid, subgrid::AbstractVector, name::String, f::Function; all=true)
    set = OrderedSet{FacetIndex}()
    for (cell_idx, cell) in enumerate(subgrid)
        for (entity_idx, entity) in enumerate(Ferrite.boundaryfunction(FacetIndex)(cell))
            pass = all
            for node_idx in entity
                v = f(get_node_coordinate(grid, node_idx))
                all ? (!v && (pass = false; break)) : (v && (pass = true; break))
            end
            pass && push!(set, FacetIndex(cell_idx, entity_idx))
        end
    end
    addfacetset!(grid, name, set)
    return grid
end
function addcustomnodeset!(grid, cellset, name, f)
    node_indices = Int64[]
    for cell in cellset
        for node_idx in cell.nodes
            f(get_node_coordinate(grid, node_idx)) && push!(node_indices, node_idx)
        end
    end
    addnodeset!(grid, name, unique(node_indices))
end

using QuadGK
function bisect(f, θ_lo, θ_hi; tolerance=1e-8)
    # bisection
    θ_mid = (θ_lo + θ_hi) / 2 # initial guess
    while θ_hi - θ_lo > tolerance
        θ_mid = (θ_lo + θ_hi) / 2
        f(θ_mid) * f(θ_lo) < 0 ? (θ_hi = θ_mid) : (θ_lo = θ_mid)
    end
    return θ_mid
end
function find_points(x, y, A, B, L)
    N = length(x)
    x_new = similar(x)
    y_new = similar(y)

    x_min = minimum(x)
    for i in (1, N)
        θ = (x[i] - x_min) * π / L
        x_new[i] = -A * cos(θ)
        y_new[i] = -B * sin(θ)
    end
    lengths = @views sqrt.((x[2:end] .- x[1:end-1]) .^ 2 .+ (y[2:end] .- y[1:end-1]) .^ 2)
    θ0 = 0.0
    for i in 1:N-2
        x0, y0, d = x_new[N-i+1], y_new[N-i+1], lengths[N-i]
        θ0 = bisect(θ0, π) do θ
            sqrt((A * cos(θ) - x0)^2 + (B * sin(θ) - y0)^2) - d
        end
        x_new[N-i] = A * cos(θ0)
        y_new[N-i] = B * sin(θ0)
    end
    x_new, y_new
end
function map_initial(x, y, Ar)
    L = maximum(x) - minimum(x)
    # find the minor/major axis that result in this length
    ds(θ, a) = sqrt(a^2 * sin(θ)^2 + (a / Ar)^2 * cos(θ)^2)
    function find_a(a)
        quadgk(θ -> ds(θ, a), 0, π)[1] - L
    end
    a0 = bisect(find_a, 0.0, L)
    a = bisect(0.98 * a0, 1.08 * a0) do a
        xi, yi = find_points(x, y, a, a / Ar, L)
        @views sum(sqrt.((xi[2:end] .- xi[1:end-1]) .^ 2 .+ (yi[2:end] .- yi[1:end-1]) .^ 2)) - L
    end
    find_points(x, y, a, a / Ar, L)
end



# Assemble K_int, R_int, K_pres and F_p (all for unit pressure p=1) in one cell loop.
function assemble_all!(K_int, r_int, K_pres, F_p, dh, scv, u, mat)
    n_e = ndofs_per_cell(dh)
    ke_i = zeros(n_e, n_e)
    re_i = zeros(n_e)
    ke_p = zeros(n_e, n_e)
    re_p = zeros(n_e)
    asm_i = start_assemble(K_int, r_int)
    asm_p = start_assemble(K_pres)
    fill!(F_p, 0.0)
    for cell in CellIterator(dh)
        fill!(ke_i, 0.0)
        fill!(re_i, 0.0)
        # fill!(ke_p, 0.0); fill!(re_p, 0.0)
        reinit!(scv, cell)
        sd = shelldofs(cell)
        u_e = u[sd]
        FerriteShells.membrane_tangent_RM_impl!(ke_i, scv, u_e, mat)
        FerriteShells.membrane_residuals_RM_impl!(re_i, scv, u_e, mat)
        bending_tangent_RM!(ke_i, scv, u_e, mat)
        bending_residuals_RM!(re_i, scv, u_e, mat)
        assemble!(asm_i, sd, ke_i, re_i)
    end
    # assemble the pressure for each cell set
    for cellset in keys(Ferrite.getcellsets(grid))
        for cell in CellIterator(dh, Ferrite.getcellset(grid, cellset))
            fill!(ke_p, 0.0)
            fill!(re_p, 0.0)
            reinit!(scv, cell)
            sd = shelldofs(cell)
            u_e = u[sd]
            assemble_pressure!(re_p, scv, u_e, pressures[cellset])
            assemble_pressure_tangent!(ke_p, scv, u_e, pressures[cellset])
            assemble!(asm_p, sd, ke_p)
            F_p[sd] .+= re_p
        end
    end
end

# load the deformed geometry
# fname = "/home/marin/Workspace/WaterLilyPreCICE/examples/0D-CalculiX/miniLIMO-0D/CalculiX/geom_deformed.inp"
fname = "/home/marin/Workspace/WaterLilyPreCICE/examples/0D-CalculiX/miniLIMO-0D/CalculiX/geom.inp"
grid = get_ferrite_grid(fname)

# boundary definition
addfacetset!(grid, "edge", x -> x[2] ≈ 0.0)
# addcustomfacetset!(grid, getcells(grid, "SRF_1"), "edge_top", x -> x[2] ≈ 0.0)
# addcustomfacetset!(grid, getcells(grid, "SRF_8"), "edge_bottom", x -> x[2] ≈ 0.0)

# Use custom NodeSet to slect prescribed Dirichlet nodes
addcustomnodeset!(grid, getcells(grid, "SRF_1"), "edge_top", x -> x[2] ≈ 0.0)
addcustomnodeset!(grid, getcells(grid, "SRF_8"), "edge_bottom", x -> x[2] ≈ 0.0)


function prescribed_u(x, t)
    return [0., 0.]
end
function prescribed_θ(x, t)
    return [0., 0.]
end

using Plots
# top_nodes = get_node_coordinate.(getnodes(grid, "edge_top"))
# idx = sortperm(top_nodes)
# L = maximum(first(maximum(top_nodes, dims=1) - minimum(top_nodes, dims=1)))
# node_sorted = top_nodes[idx]
# Ar = 80.2 / 55.2 # from Nienke
# x, y = getindex.(node_sorted, 1), getindex.(node_sorted, 2)
# x_new, y_new = map_initial(x, y, Ar)
# plot(x_new, y_new, aspect_ratio=:equal)
# scatter!(x_new, y_new)


function generate_boundary_function(grid, nodeset)
    top_nodes = get_node_coordinate.(getnodes(grid, nodeset))
    idx = sortperm(top_nodes)
    node_sorted = top_nodes[idx]
    Ar = 80.2 / 55.2 # from Nienke
    x, y = getindex.(node_sorted, 1), getindex.(node_sorted, 3)
    x_new, y_new = map_initial(x, y, Ar)
    Xs = vcat(x', y'); dXs = vcat(x_new' .- x', y_new' .- y')
    return function prescribed_u(x, t)
        idx = findmin(dropdims(sum(abs2, Xs .- [x[1], x[2]], dims=1), dims=1))[2]
        return min(t,1).*dXs[:, idx] # linear ramp
    end
end

prescribed_u_top = generate_boundary_function(grid, "edge_top")
prescribed_u_bottom = generate_boundary_function(grid, "edge_bottom")

# interpolation space and quadrature
ip = Lagrange{RefQuadrilateral,1}()
qr = QuadratureRule{RefQuadrilateral}(3)
scv = ShellCellValues(qr, ip, ip)

# material model
mat = LinearElastic(20000.0, 0.33, 0.0016)

# degrees of freedom of the problem
dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

# make a dict of pressure and cellset
pressures = Dict("SRF_1" => 1.104, "SRF_2" => 1.104,
    "SRF_3" => 1.104, "SRF_4" => 1.104,
    "SRF_5" => -1.104, "SRF_6" => -1.104,
    "SRF_7" => -1.104, "SRF_8" => -1.104,
    "SRF_9" => -1.104, "SRF_10" => -1.104,
    "SRF_11" => -1.104, "SRF_12" => 1.104,
    "SRF_13" => 1.104, "SRF_14" => 1.104)

# add boundary conditions
ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "edge"), x -> 0.0, [2]))
add!(ch, Dirichlet(:u, getnodeset(grid, "edge_top"), (x, t) -> prescribed_u_top(x, t), [1, 3]))
# add!(ch, Dirichlet(:θ, getnodeset(grid, "edge_top"), (x, t) -> prescribed_θ(x, t), [1, 2]))
add!(ch, Dirichlet(:u, getnodeset(grid, "edge_bottom"), (x, t) -> [1,-1].*prescribed_u_bottom(x, t), [1, 3]))
# add!(ch, Dirichlet(:θ, getnodeset(grid, "edge_bottom"), (x, t) -> prescribed_θ(x, t), [1, 2]))
close!(ch); update!(ch, 1.0)

# allocate matrix and vectors
N = ndofs(dh)
K_int = allocate_matrix(dh)
K_pres = allocate_matrix(dh)
r_int = zeros(N)
f_p = zeros(N)
u = zeros(N)

# assemble
p = 1.0
@time assemble_all!(K_int, r_int, K_pres, f_p, dh, scv, u, mat)
K_eff = K_int - p .* K_pres
rhs1 = p .* f_p .- r_int
apply_zero!(K_eff, rhs1, ch) # this does nothing
apply!(K_eff, rhs1, ch)
# solve
@time u = K_eff \ rhs1
v2 = K_eff \ f_p

# save results
VTKGridFile("LIMO", dh) do vtk
    write_solution(vtk, dh, u)
    # Ferrite.write_node_data(vtk, v2, "load dir")
    Ferrite.write_constraints(vtk, ch)
    color(vtk, grid)
end
