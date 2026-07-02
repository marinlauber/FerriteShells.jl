using FerriteShells, LinearAlgebra, Printf, WriteVTK

# Assemble K_int, R_int, K_plv and F_plv (all for unit pressure p=1) in one cell loop.
function assemble_all!(K_int, r_int, dh, scv, u, mat)
    n_e = ndofs_per_cell(dh)
    ke_i = zeros(n_e, n_e); re_i = zeros(n_e)
    asm_i = start_assemble(K_int, r_int)
    for cell in CellIterator(dh)
        fill!(ke_i, 0.0); fill!(re_i, 0.0)
        reinit!(scv, cell)
        sd  = shelldofs(cell)
        u_e = u[sd]
        membrane_residuals_RM!(re_i, scv, u_e, mat)
        bending_residuals_RM!(re_i, scv, u_e, mat)
        membrane_tangent_RM!(ke_i, scv, u_e, mat)
        bending_tangent_RM!(ke_i, scv, u_e, mat)
        assemble!(asm_i, sd, ke_i, re_i)
    end
end

function assemble_pressure_region!(K_i, F_i, scv, u_vec, dh)
    n_e = ndofs_per_cell(dh)
    ke_p = zeros(n_e, n_e)
    re_p = zeros(n_e)
    asm_p = start_assemble(K_i)
    for cell in CellIterator(dh)
        fill!(ke_p, 0.0); fill!(re_p, 0.0)
        reinit!(scv, cell)
        sd = shelldofs(cell)
        u_e = u_vec[sd]
        assemble_pressure!(re_p, scv, u_e, 1.0) # unit pressure
        assemble_pressure_tangent!(ke_p, scv, u_e, 1.0)
        assemble!(asm_p, sd, ke_p)
        F_i[sd] .+= re_p
    end
end

# Maps an n×n structured grid on [-1,1]² to a disk of radius R via the
# Shirley–Chiu concentric mapping.  All elements are quadrilaterals; no
# degenerate (collapsed) elements at the centre.
# primitive=Quadrilateral → (n+1)²  nodes, n² Q4 cells
# primitive=QuadraticQuadrilateral → (2n+1)² nodes, n² Q9 cells (midpoints on disk surface)
function make_circular_plate_grid(R, n; primitive=Quadrilateral)
    function sq2disk(ξ, η)
        iszero(ξ) && iszero(η) && return (0.0, 0.0)
        if abs(ξ) >= abs(η)
            r = ξ;  θ = π * η / (4ξ)
        else
            r = η;  θ = π/2 - π * ξ / (4η)
        end
        (R * r * cos(θ), R * r * sin(θ))
    end
    if primitive === Quadrilateral
        nodes = [Vec{2}(sq2disk(-1.0 + 2i/n, -1.0 + 2j/n))
                 for j in 0:n for i in 0:n]
        nid = (i, j) -> j*(n+1) + i + 1
        cells = [Quadrilateral((nid(i,j), nid(i+1,j), nid(i+1,j+1), nid(i,j+1)))
                 for j in 0:n-1 for i in 0:n-1]
    else  # QuadraticQuadrilateral
        nodes = [Vec{2}(sq2disk(-1.0 + i/n, -1.0 + j/n))
                 for j in 0:2n for i in 0:2n]
        nid = (i, j) -> j*(2n+1) + i + 1
        cells = [QuadraticQuadrilateral((
                     nid(2i,   2j),   nid(2i+2, 2j),
                     nid(2i+2, 2j+2), nid(2i,   2j+2),
                     nid(2i+1, 2j),   nid(2i+2, 2j+1),
                     nid(2i+1, 2j+2), nid(2i,   2j+1),
                     nid(2i+1, 2j+1)))
                 for j in 0:n-1 for i in 0:n-1]
    end
    grid = shell_grid(Grid(cells, Node.(nodes)))   # embed 2D→3D via shell_grid
    addnodeset!(grid, "boundary", x -> norm(x) ≈ R)
    return grid
end

# make the mesh
grid = make_circular_plate_grid(7.5, 16)

# interpolation space
ip  = Lagrange{RefQuadrilateral,1}()
qr  = QuadratureRule{RefQuadrilateral}(2)
fqr = FacetQuadratureRule{RefQuadrilateral}(2)
scv = ShellCellValues(qr, ip, ip; mitc=MITC4)

# Mooney–Rivlin material
C₁, C₂, t, p_max = 80.0, 20.0, 0.84, 35.0
mat = Hyperelastic(C -> (
    I₁ = tr(C); I₂ = (I₁*I₁ - C⊡C)/2.0;
    C₁*(I₁-3) + C₂*(I₂-3)), t
)
# mat = LinearElastic(C₁, 0.3, t)

# DOF handler (must precede ConstraintHandler)
dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

# boundary conditions - fixed outer edge
ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getnodeset(grid, "boundary"), x -> zero(x), [1,2,3]))
close!(ch); Ferrite.update!(ch, 0.0)

let
# Matrices and vectors
N     = ndofs(dh)
K_int = allocate_matrix(dh)
K_ext = allocate_matrix(dh)
K_eff = allocate_matrix(dh)
r_int = zeros(N)
f_ext = zeros(N)
f_ref = zeros(N)
rhs   = zeros(N)
u     = zeros(N)
Δu    = zeros(N)
v1    = zeros(N)
v2    = zeros(N)

# Initial symbolic LU — structure is reused via lu!/ldiv! throughout
assemble_all!(K_int, r_int, dh, scv, u, mat)
K_eff.nzval .= K_int.nzval
apply_zero!(K_eff, r_int, ch)
F_lu = lu(K_eff)
free = ch.free_dofs

# VTK output
pvd      = paraview_collection("circular-hyperelastic-plate")
vtk_step = Ref(0)

# Cylindrical arc-length (Riks) solver
# Equilibrium:   r_int(u) − λ·p_max·F_ext(u) = 0
# AL constraint: ‖Δu[free]‖² = Δs²  (cylindrical, ψ = 0)
Δs       = 1e-2
n_steps  = 1000
max_iter = 20
tol_nl   = 1e-6

λ      = 0.0
Δu_dir = zeros(N)   # step direction from last converged step
Δλ_dir = 1.0        # seed: start by increasing load

println("  step |    λ    |   p [Pa] | iters")
for step in 1:n_steps
    u_n = copy(u);  λ_n = λ

    # Predictor: tangent direction at (u_n, λ_n)
    assemble_all!(K_int, r_int, dh, scv, u_n, mat)
    fill!(f_ext, 0.0)
    assemble_pressure_region!(K_ext, f_ext, scv, u_n, dh)
    K_eff.nzval .= K_int.nzval .- λ_n .* p_max .* K_ext.nzval
    f_ref .= p_max .* f_ext;  f_ref[ch.prescribed_dofs] .= 0.0
    apply_zero!(K_eff, r_int, ch)        # sets BC rows/cols of K_eff to identity
    lu!(F_lu, K_eff)
    ldiv!(v2, F_lu, f_ref)               # v2 = K_eff⁻¹ · p_max·F_ext (load tangent)

    s       = dot(Δu_dir[free], v2[free]) + Δλ_dir   # sign from previous direction
    δλ_pred = copysign(Δs / sqrt(dot(v2[free], v2[free])), s)
    u  .= u_n .+ δλ_pred .* v2;  apply!(u, ch)
    λ   = λ_n + δλ_pred
    Δu .= u .- u_n;  Δλ = δλ_pred

    # Corrector: Newton on the augmented (u, λ) system
    converged = false;  n_iter = 0
    for iter in 1:max_iter
        assemble_all!(K_int, r_int, dh, scv, u, mat)
        fill!(f_ext, 0.0)
        assemble_pressure_region!(K_ext, f_ext, scv, u, dh)
        K_eff.nzval .= K_int.nzval .- λ .* p_max .* K_ext.nzval

        rhs .= λ .* p_max .* f_ext .- r_int
        apply_zero!(K_eff, rhs, ch)
        norm(rhs[free]) < tol_nl && (converged = true; n_iter = iter - 1; break)
        n_iter = iter

        lu!(F_lu, K_eff)
        ldiv!(v1, F_lu, rhs)             # v1 = K_eff⁻¹ · (λ·p·F_ext − R_int)
        f_ref .= p_max .* f_ext;  f_ref[ch.prescribed_dofs] .= 0.0
        ldiv!(v2, F_lu, f_ref)           # v2 = K_eff⁻¹ · p_max·F_ext (load mode)

        # cylindrical correction: 2·Δu·δu = −(‖Δu‖² − Δs²)
        g  = dot(Δu[free], Δu[free]) - Δs^2
        δλ = (-g/2 - dot(Δu[free], v1[free])) / dot(Δu[free], v2[free])
        v1 .+= δλ .* v2                  # v1 = δu
        u  .+= v1;  apply!(u, ch)
        λ  += δλ
        Δu .+= v1;  Δλ += δλ
    end

    if !converged
        @warn "arc-length step $step (λ=$(round(λ; digits=4))) did not converge; aborting"
        break
    end

    Δu_dir .= Δu;  Δλ_dir = Δλ   # save direction for next predictor sign

    VTKGridFile("circular-hyperelastic-plate-$(vtk_step[])", dh) do vtk
        vtk_step[] += 1
        write_solution(vtk, dh, u)
        pvd[vtk_step[]] = vtk
    end
    @printf("  %4d |  %7.4f |  %8.3f |  %d\n", step, λ, λ * p_max, n_iter)

    λ ≥ 1.0 && break
end
vtk_save(pvd)
end # let