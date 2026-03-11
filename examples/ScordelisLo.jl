using FerriteShells

# Scordelis-Lo roof — Kirchhoff-Love shell
#
# Cylindrical barrel vault: R=25, L=50, half-angle Φ=40°, t=0.25, E=4.32e8, ν=0.
# Self-weight q=90 per unit area acts in the -y direction (gravity).
# BCs: rigid diaphragm at the straight ends (x=0, x=L) → u_y=u_z=0; free curved edges.
# Reference vertical deflection at midpoint of free edge: 0.3024.
#
# Coordinate map: 2D grid (θ,axial) → (axial, R·cos θ, R·sin θ)
#   Crown (θ=0):      X = (axial, R, 0)          — top of vault
#   Free edges (θ=±Φ): X = (axial, R·cosΦ, ±R·sinΦ) — lower edges

const R_sl, L_sl, Φ_sl = 25.0, 50.0, 40π/180
const E_sl, ν_sl, t_sl = 4.32e8, 0.0, 0.25
const q_sl = Vec{3}((0.0, -90.0, 0.0))  # self-weight in -y direction

function scordelis_lo_grid(ns, nt)
    g = shell_grid(
        generate_grid(QuadraticQuadrilateral, (ns, nt),
                      Vec{2}((-Φ_sl, 0.0)), Vec{2}((Φ_sl, L_sl)));
        map = n -> (n.x[2], R_sl * cos(n.x[1]), R_sl * sin(n.x[1])))
    addnodeset!(g, "diaphragm", x -> x[1] ≈ 0.0 || x[1] ≈ L_sl)
    addnodeset!(g, "ref_point",
        x -> abs(x[1] - L_sl/2) < 1e-8 && abs(x[2] - R_sl*cos(Φ_sl)) < 1e-8 &&
             abs(x[3] - R_sl*sin(Φ_sl)) < 1e-8)
    return g
end

function scordelis_lo_solve(ns, nt)
    ip  = Lagrange{RefQuadrilateral, 2}()
    qr  = QuadratureRule{RefQuadrilateral}(4)
    scv = ShellCellValues(qr, ip, ip)
    mat = LinearElastic(E_sl, ν_sl, t_sl)

    grid   = scordelis_lo_grid(ns, nt)
    dh     = DofHandler(grid); add!(dh, :u, ip^3); close!(dh)
    n_el   = ndofs_per_cell(dh)
    n_base = getnbasefunctions(ip)

    K  = allocate_matrix(dh)
    f  = zeros(ndofs(dh))
    asmb = start_assemble(K, zeros(ndofs(dh)))
    ke = zeros(n_el, n_el); re = zeros(n_el); fe = zeros(n_el)

    for cell in CellIterator(dh)
        fill!(ke, 0.0); fill!(re, 0.0); fill!(fe, 0.0)
        reinit!(scv, cell)
        u0 = zeros(n_el)
        membrane_tangent_KL!(ke, scv, u0, mat)
        bending_tangent_KL!(ke, scv, u0, mat)
        assemble!(asmb, celldofs(cell), ke, re)
        for qp in 1:getnquadpoints(scv)
            ξ  = scv.qr.points[qp]; dΩ = scv.detJdV[qp]
            for I in 1:n_base
                NI = Ferrite.reference_shape_value(ip, ξ, I)
                @views fe[3I-2:3I] .+= NI * q_sl * dΩ
            end
        end
        @views f[celldofs(cell)] .+= fe
    end

    dbc = ConstraintHandler(dh)
    add!(dbc, Dirichlet(:u, getnodeset(grid, "diaphragm"), x -> zeros(2), [2, 3]))
    close!(dbc); Ferrite.update!(dbc, 0.0); apply!(K, f, dbc)

    u_sol = K \ f

    ref_nodes = collect(getnodeset(grid, "ref_point"))
    @assert length(ref_nodes) == 1 "Expected exactly one reference node"
    for cell in CellIterator(dh)
        for (I, gid) in enumerate(getnodes(cell))
            if gid == ref_nodes[1]
                cd = celldofs(cell)
                return u_sol[cd[3I-1]]  # y-component (vertical)
            end
        end
    end
    error("ref_point node not found in any cell")
end

w = scordelis_lo_solve(16, 16)
println("Scordelis-Lo (KL, 16×16): u_y at free-edge midpoint = $(round(w; digits=5))  (reference: -0.3024)")
