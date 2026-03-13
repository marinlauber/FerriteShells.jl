using FerriteShells
using LinearAlgebra

function make_quarter_pillow_grid(n; L=1.0)
    corners = [Vec{2}((0.0, 0.0)), Vec{2}((L/2, 0.0)), Vec{2}((L/2, L/2)), Vec{2}((0.0, L/2))]
    grid = shell_grid(generate_grid(QuadraticQuadrilateral, (n, n), corners))
    addnodeset!(grid, "edge",  x -> isapprox(x[1], L/2, atol=1e-10) || isapprox(x[2], L/2, atol=1e-10))
    addnodeset!(grid, "sym_x", x -> isapprox(x[1], 0.0,  atol=1e-10))
    addnodeset!(grid, "sym_y", x -> isapprox(x[2], 0.0,  atol=1e-10))
    return grid
end

n=4; L=1.0; mat=LinearElastic(1.0e6, 0.3, 1e-3)
grid = make_quarter_pillow_grid(n; L)
ip = Lagrange{RefQuadrilateral, 2}()
qr = QuadratureRule{RefQuadrilateral}(3)
scv = ShellCellValues(qr, ip, ip)
dh = DofHandler(grid); add!(dh, :u, ip^3); close!(dh)
ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getnodeset(grid, "edge"), x -> 0.0, [3]))
add!(ch, Dirichlet(:u, getnodeset(grid, "sym_x"), x -> 0.0, [1]))
add!(ch, Dirichlet(:u, getnodeset(grid, "sym_y"), x -> 0.0, [2]))
close!(ch); Ferrite.update!(ch, 0.0)

u = zeros(ndofs(dh))
for cell in CellIterator(dh)
    x = getcoordinates(cell); dofs = celldofs(cell)
    for i in eachindex(x)
        u[dofs[3i]] = 1e-2 * sin(π * x[i][1] / (L/2)) * sin(π * x[i][2] / (L/2))
    end
end
apply!(u, ch)

println("ndofs = ", ndofs(dh), ", max |u_z| = ", maximum(abs, u[3:3:end]))

N = ndofs(dh); p = 10.0
K_mem = allocate_matrix(dh); r_int = zeros(N); F_ext = zeros(N); K_pres = allocate_matrix(dh)

n_e = ndofs_per_cell(dh)
ke = zeros(n_e, n_e); re = zeros(n_e)
asm = start_assemble(K_mem, r_int)
for cell in CellIterator(dh)
    fill!(ke, 0.0); fill!(re, 0.0)
    reinit!(scv, cell)
    x = getcoordinates(cell); u_e = u[celldofs(cell)]
    membrane_tangent_KL!(ke, scv, x, u_e, mat)
    membrane_residuals_KL!(re, scv, x, u_e, mat)
    bending_tangent_KL!(ke, scv, x, u_e, mat)
    bending_residuals_KL!(re, scv, x, u_e, mat)
    assemble!(asm, celldofs(cell), ke, re)
end

re2 = zeros(ndofs_per_cell(dh))
for cell in CellIterator(dh)
    fill!(re2, 0.0); reinit!(scv, cell)
    x = getcoordinates(cell); u_e = u[celldofs(cell)]
    assemble_pressure!(re2, scv, x, u_e, p)
    F_ext[celldofs(cell)] .+= re2
end

ke2 = zeros(n_e, n_e); asm2 = start_assemble(K_pres)
for cell in CellIterator(dh)
    fill!(ke2, 0.0); reinit!(scv, cell)
    x = getcoordinates(cell); u_e = u[celldofs(cell)]
    assemble_pressure_tangent!(ke2, scv, x, u_e, p)
    assemble!(asm2, celldofs(cell), ke2)
end

rhs = F_ext - r_int
K_eff = K_mem - K_pres

println("||r_int||  = ", norm(r_int))
println("||F_ext||  = ", norm(F_ext))
println("||rhs before apply|| = ", norm(rhs))

apply_zero!(K_eff, rhs, ch)
println("||rhs after apply_zero|| = ", norm(rhs))

λ = eigvals(Symmetric(Matrix(K_eff)))
println("min eigval K_eff = ", minimum(λ))
println("max eigval K_eff = ", maximum(λ))
println("n_negative = ", count(<(0), λ))

du = K_eff \ rhs
println("||du||     = ", norm(du))
println("max |du|   = ", maximum(abs, du))
println("After step, max u_z = ", maximum(u[3:3:end] .+ du[3:3:end]))
println("After step, min u_z = ", minimum(u[3:3:end] .+ du[3:3:end]))

# Now simulate 5 iterations
u2 = copy(u)
for iter in 1:5
    fill!(F_ext, 0.0); fill!(r_int, 0.0)
    asm = start_assemble(K_mem, r_int)
    for cell in CellIterator(dh)
        fill!(ke, 0.0); fill!(re, 0.0); reinit!(scv, cell)
        x = getcoordinates(cell); u_e = u2[celldofs(cell)]
        membrane_tangent_KL!(ke, scv, x, u_e, mat); membrane_residuals_KL!(re, scv, x, u_e, mat)
        bending_tangent_KL!(ke, scv, x, u_e, mat); bending_residuals_KL!(re, scv, x, u_e, mat)
        assemble!(asm, celldofs(cell), ke, re)
    end
    for cell in CellIterator(dh)
        fill!(re2, 0.0); reinit!(scv, cell)
        x = getcoordinates(cell); u_e = u2[celldofs(cell)]
        assemble_pressure!(re2, scv, x, u_e, p); F_ext[celldofs(cell)] .+= re2
    end
    asm2 = start_assemble(K_pres)
    for cell in CellIterator(dh)
        fill!(ke2, 0.0); reinit!(scv, cell)
        x = getcoordinates(cell); u_e = u2[celldofs(cell)]
        assemble_pressure_tangent!(ke2, scv, x, u_e, p); assemble!(asm2, celldofs(cell), ke2)
    end
    K_eff2 = K_mem - K_pres
    rhs2 = F_ext - r_int
    apply_zero!(K_eff2, rhs2, ch)
    println("iter $iter: ||rhs|| = $(norm(rhs2))")
    if norm(rhs2) < 1e-6; println("converged!"); break; end
    u2 .+= K_eff2 \ rhs2
    println("  max u_z = $(maximum(u2[3:3:end])), min u_y = $(minimum(u2[2:3:end]))")
end
