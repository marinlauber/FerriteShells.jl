using FerriteShells

# Scordelis-Lo roof — Reissner-Mindlin shell
# Same geometry and loading as ScordelisLo.jl; see that file for full description.
# Two-field DofHandler (:u ip^3, :θ ip^2).  DOF reordering via shelldofs().

const R_sl, L_sl, Φ_sl = 25.0, 50.0, 40π/180
const E_sl, ν_sl, t_sl = 4.32e8, 0.0, 0.25
const q_sl = Vec{3}((0.0, -90.0, 0.0))

function scordelis_lo_rm_grid(ns, nt)
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

function scordelis_lo_rm_solve(ns, nt)
    ip  = Lagrange{RefQuadrilateral, 2}()
    qr  = QuadratureRule{RefQuadrilateral}(4)
    scv = ShellCellValues(qr, ip, ip)
    mat = LinearElastic(E_sl, ν_sl, t_sl)

    grid = scordelis_lo_rm_grid(ns, nt)
    dh   = DofHandler(grid); add!(dh, :u, ip^3); add!(dh, :θ, ip^2); close!(dh)
    n_el   = ndofs_per_cell(dh)   # 5·n_base (interleaved after shelldofs)
    n_base = getnbasefunctions(ip)

    K  = allocate_matrix(dh)
    f  = zeros(ndofs(dh))
    asmb = start_assemble(K, zeros(ndofs(dh)))
    ke = zeros(5n_base, 5n_base); re = zeros(5n_base); fe = zeros(5n_base)

    for cell in CellIterator(dh)
        fill!(ke, 0.0); fill!(re, 0.0); fill!(fe, 0.0)
        reinit!(scv, cell)
        x  = getcoordinates(cell)
        u0 = zeros(5n_base)
        membrane_tangent_RM!(ke, scv, x, u0, mat)
        bending_tangent_RM!(ke, scv, x, u0, mat)
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
                return u_sol[cd[3I-1]]  # y-component of :u
            end
        end
    end
    error("ref_point node not found in any cell")
end

w = scordelis_lo_rm_solve(16, 16)
println("Scordelis-Lo (RM, 16×16): u_y at free-edge midpoint = $(round(w; digits=5))  (reference: -0.3024)")
