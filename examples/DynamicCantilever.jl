using FerriteShells,LinearAlgebra,Printf

# assembly functions
function assemble_shell!(K, r, dh, scv, u, mat)
    n_e = ndofs_per_cell(dh)
    ke  = zeros(n_e, n_e); re = zeros(n_e)
    asm = start_assemble(K, r)
    for cell in CellIterator(dh)
        fill!(ke, 0.0); fill!(re, 0.0)
        reinit!(scv, cell)
        sd  = shelldofs(cell)
        u_e = u[sd]
        membrane_residuals_RM!(re, scv, u_e, mat)
        bending_residuals_RM!(re, scv, u_e, mat)
        membrane_tangent_RM!(ke, scv, u_e, mat)
        bending_tangent_RM!(ke, scv, u_e, mat)
        assemble!(asm, sd, ke, re)
    end
end
function assemble_mass_global!(M, dh, scv, ρ, mat)
    n_e = ndofs_per_cell(dh)
    me  = zeros(n_e, n_e)
    asm = start_assemble(M)
    for cell in CellIterator(dh)
        fill!(me, 0.0)
        reinit!(scv, cell)
        mass_matrix!(me, scv, ρ, mat)
        assemble!(asm, shelldofs(cell), me)
    end
end

# mesh dimensions
Lx  = 10.0      # length [m]
Ly  = 1.0       # width [m]
ts  = 0.1       # shell thickness [m]
E   = 1.2e6     # Young's modulus [Pa]
ν   = 0.0       # Poisson ratio (decouples bending/membrane for easy comparison)
ρ   = 1.0       # density [kg/m³]
q_z = 0.06      # tip traction [N/m] (distributed over width Ly)

# material and thickness
mat = LinearElastic(E, ν, ts)

# Analytical reference: Euler-Bernoulli cantilever, first bending mode
EI_b  = E * Ly * ts^3 / 12          # bending stiffness [N·m²]
ρA_b  = ρ * Ly * ts                 # mass per unit length [kg/m]
F_tip = q_z * Ly                    # total tip force [N]
u_s   = F_tip * Lx^3 / (3EI_b)     # static tip deflection [m]
ω₁    = 3.516 / Lx^2 * sqrt(EI_b / ρA_b)   # fundamental frequency [rad/s]
T₁    = 2π / ω₁                    # period [s]

# make the grid
grid = shell_grid(generate_grid(QuadraticQuadrilateral, (16, 2), Vec{2}((0.0, 0.0)), Vec{2}((Lx, Ly))))
addfacetset!(grid, "clamped", x -> isapprox(x[1], 0.0,  atol=1e-10))
addfacetset!(grid, "tip",     x -> isapprox(x[1], Lx,   atol=1e-10))
addnodeset!(grid,  "tip_mid", x -> isapprox(x[1], Lx,   atol=1e-10) &&
                                   isapprox(x[2], Ly/2, atol=1e-10))

# interpolation space and shell
ip  = Lagrange{RefQuadrilateral, 2}()
qr  = QuadratureRule{RefQuadrilateral}(3)
fqr = FacetQuadratureRule{RefQuadrilateral}(3)
scv = ShellCellValues(qr, ip, ip; mitc=MITC9)

# degrees of freedom of the problem
dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)
ndof = ndofs(dh)

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "clamped"), x -> zeros(3), [1,2,3]))
add!(ch, Dirichlet(:θ, getfacetset(grid, "clamped"), x -> zeros(2), [1,2]))
close!(ch)
Ferrite.update!(ch, 0.0)
free = ch.free_dofs

# Find u_z DOF at tip midpoint via a unit probe load
f_probe = zeros(ndof)
apply_pointload!(f_probe, dh, "tip_mid", Vec{3}((0.0, 0.0, 1.0)))
w_tip = argmax(f_probe)

# allocate matrices and vectors
K_int = allocate_matrix(dh)
K_eff = allocate_matrix(dh)
M     = allocate_matrix(dh)
u     = zeros(ndof); apply!(u, ch)
v     = zeros(ndof)
a     = zeros(ndof)
r_int = zeros(ndof)
f_ext = zeros(ndof)
r_old = zeros(ndof)   # r_int(u_n), for α·r_old in HHT residual
δu    = zeros(ndof)
rhs   = zeros(ndof)

# assemble mass once, stays the same
assemble_mass_global!(M, dh, scv, ρ, mat)
assemble_traction!(f_ext, dh, getfacetset(grid, "tip"), ip, fqr, Vec{3}((0.0, 0.0, q_z)))

# HHT-α parameters
α_hht = -0.05
γ_hht = 0.5 - α_hht
β_hht = (1 - α_hht)^2 / 4
Δt      = T₁ / 50
n_steps = round(Int, 2.5 * T₁ / Δt)
tol_dyn = 1e-8
max_iter = 10

# Initial K_eff for symbolic LU factorization (pattern setup)
fill!(K_int, 0.0); fill!(r_int, 0.0)
assemble_shell!(K_int, r_int, dh, scv, zeros(ndof), mat)
K_eff.nzval .= M.nzval ./ (β_hht * Δt^2) .+ (1 - α_hht) .* K_int.nzval
rhs_dummy = zeros(ndof)
apply_zero!(K_eff, rhs_dummy, ch)
F_lu = lu(K_eff)

println("Static tip deflection:  u_s = $(round(u_s, digits=4)) m")
println("Fundamental period:     T₁  = $(round(T₁,  digits=3)) s")
println("Time step:              Δt  = $(round(Δt,  digits=4)) s  (T₁/50)\n")
@printf("%-8s  %-12s  %-12s  %-6s\n", "t [s]", "u_z [m]", "u_anal [m]", "iters")

using WriteVTK
pvd = paraview_collection("cantilever_dynamic")
vtk_step = Ref(0)
# save to the pvd
VTKGridFile("cantilever_dynamic-$(vtk_step[])", dh) do vtk
    write_solution(vtk, dh, u); pvd[0.0] = vtk
end

# time integration
for step in 1:n_steps
    t_now = step * Δt
    # Predictor: advance kinematics without equilibrium correction
    ũ = u .+ Δt .* v .+ (Δt^2 * (0.5 - β_hht)) .* a
    ṽ = v .+ (Δt  * (1 - γ_hht)) .* a
    # initial guess
    u_new = copy(ũ)
    apply!(u_new, ch)
    converged = false; iters = 0; vtk_step[] += 1
    # Newton-Raphson loop
    for iter in 1:max_iter
        iters = iter

        fill!(K_int, 0.0); fill!(r_int, 0.0)
        assemble_shell!(K_int, r_int, dh, scv, u_new, mat)

        # Acceleration from Newmark relation
        a_new = (u_new .- ũ) ./ (β_hht * Δt^2)

        # HHT residual: R = M·ä + (1-α)·r_int(u_{n+1}) + α·r_int(u_n) − f_ext
        rhs .= -(M * a_new .+ (1 - α_hht) .* r_int .+ α_hht .* r_old .- f_ext)
        apply_zero!(rhs, ch)

        norm(@views rhs[free]) < tol_dyn && (converged = true; break)

        # Effective stiffness and Newton update
        K_eff.nzval .= M.nzval ./ (β_hht * Δt^2) .+ (1 - α_hht) .* K_int.nzval
        apply_zero!(K_eff, rhs, ch)
        lu!(F_lu, K_eff)
        ldiv!(δu, F_lu, rhs)
        u_new .+= δu
        apply!(u_new, ch)
    end

    !converged && @warn "Step $step (t=$(round(t_now, digits=3)) s): no convergence in $max_iter iters"

    # Update state vectors
    a     .= (u_new .- ũ) ./ (β_hht * Δt^2)
    v     .= ṽ .+ (Δt * γ_hht) .* a
    r_old .= r_int
    u     .= u_new

    # save to the pvd
    VTKGridFile("cantilever_dynamic-$(vtk_step[])", dh) do vtk
        write_solution(vtk, dh, u); pvd[t_now] = vtk
    end

    # Compare with analytical first-mode response under step load
    u_anal = u_s * (1 - cos(ω₁ * t_now))
    step % 5 == 0 && @printf("%-8.3f  %-12.6f  %-12.6f  %-6d\n", t_now, u[w_tip], u_anal, iters)
end
close(pvd);