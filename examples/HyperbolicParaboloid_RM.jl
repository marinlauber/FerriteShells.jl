using FerriteShells, LinearAlgebra, Printf
using WriteVTK

# Partly clamped hyperbolic paraboloid — Lee & Bathe, Comp. Struct. 80 (2002) 235-255
# Bending-dominated benchmark (Section 3.3, Fig. 17).
# Surface: z = x² - y², (x,y) ∈ [-L/2,L/2]², clamped at y = -L/2, free elsewhere.
# Loading: self-weight q = 80·q₀(ε,μ) per unit area (Bathe Eq. 23, F∝ε scaling).
# Table 2 gives q₀(ε=0.01, μ=1) = 1.0, so q = 80 [force/area] for t/L=0.01.
# Reference scaled strain energy E₀ = E_actual/q₀ (half-shell, Bathe Table 7):
#   t/L = 0.01  →  E₀ = 8.37658e-4  →  E_actual_half = 8.37658e-4
#   t/L = 0.001 →  E₀ = 5.48614e-2  →  q₀ = 0.1  →  E_actual_half = 5.486e-3
# Full-shell strain energy = 2 × E_actual_half (using x-symmetry of the surface).

const L  = 1.0
const E  = 2e11
const ν  = 0.3
const tL = 0.01         # thickness ratio t/L
const t  = tL * L
const q_z = 80.0       # self-weight load per unit area (q₀=1 at ε=0.01, Bathe Table 2)

mat  = LinearElastic(E, ν, t)
grid = shell_grid(generate_grid(QuadraticQuadrilateral, (200, 200),
                                Vec(-L/2, -L/2), Vec(L/2, L/2));
                  map = n -> (n.x[1], n.x[2], n.x[1]^2 - n.x[2]^2))

addfacetset!(grid, "clamped", x -> x[2] ≈ -L/2)

ip  = Lagrange{RefQuadrilateral, 2}()
qr  = QuadratureRule{RefQuadrilateral}(3)
nf  = NodeFrames(grid, ip)
scv = ShellCellValues(qr, ip, ip; mitc=MITC9)

dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

dbc = ConstraintHandler(dh)
add!(dbc, Dirichlet(:u, getfacetset(grid, "clamped"), x -> zero(x), [1,2,3]))
add!(dbc, Dirichlet(:θ, getfacetset(grid, "clamped"), x -> zeros(2), [1,2]))
close!(dbc)

n_el   = ndofs_per_cell(dh)
n_base = getnbasefunctions(scv.ip_shape)
ke = zeros(n_el, n_el)
fe = zeros(n_el)

K     = allocate_matrix(dh)
f_ext = zeros(ndofs(dh))

asm = start_assemble(K, f_ext)
for cell in CellIterator(dh)
    fill!(ke, 0.0); fill!(fe, 0.0)
    reinit!(scv, cell, nf)
    u_e = zeros(n_el)
    membrane_tangent_RM!(ke, scv, u_e, mat)
    bending_tangent_RM!(ke, scv, u_e, mat)
    for qp in 1:getnquadpoints(scv)
        dΩ = scv.detJdV[qp]; ξ = scv.qr.points[qp]
        for I in 1:n_base
            NI = Ferrite.reference_shape_value(scv.ip_shape, ξ, I)
            fe[5(I-1)+3] -= NI * q_z * dΩ   # u_z in shelldofs ordering
        end
    end
    assemble!(asm, shelldofs(cell), ke, fe)
end

f_ref = copy(f_ext)   # save before BCs overwrite f_ext
apply!(K, f_ext, dbc)
u = K \ f_ext

E_total = 0.5 * dot(u, f_ref)                # = 0.5 · f_ext · u at equilibrium
E_ref   = 2 * 8.37658e-4                     # full shell: 2 × E₀ (q₀=1 at ε=0.01), Bathe Table 7

@printf("Strain energy (computed) : %.5e\n", E_total)
@printf("Strain energy (Bathe ref): %.5e\n", E_ref)
@printf("Error                    : %.2f%%\n", abs(E_total - E_ref) / E_ref * 100)

VTKGridFile("hyperbolic_paraboloid", dh) do vtk
    write_solution(vtk, dh, u)
end
