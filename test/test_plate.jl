using FerriteShells
using LinearAlgebra
using Test

@testset "KL bending energy h-convergence" begin
    # For w = sin(πx)sin(πy) on [0,1]², the exact KL bending energy is W = ½Dπ⁴.
    # Proof: κ₁₁=κ₂₂=-π²sinsinand κ₁₂=π²coscos; each squared integral over [0,1]² = π⁴/4;
    # contracting with D^{αβγδ} gives sum = 4Et/(1-ν²), then ×(t²/12)×(½)×(π⁴/4)×4 = ½Dπ⁴.
    # We project the exact mode onto Q9 nodes and verify the assembled FEM energy
    # converges to W_exact at rate ≥ 2 (Q9 H²-interpolation error is O(h)).

    E, ν, t = 1e4, 0.3, 0.01
    D       = E * t^3 / (12 * (1 - ν^2))
    W_exact = 0.5 * D * π^4

    function kl_bending_energy_fem(n)
        ip  = Lagrange{RefQuadrilateral, 2}()
        qr  = QuadratureRule{RefQuadrilateral}(4)
        scv = ShellCellValues(qr, ip, ip)
        mat = LinearElastic(E, ν, t)
        grid = shell_grid(generate_grid(QuadraticQuadrilateral, (n, n),
                                        Vec{2}((0.0, 0.0)), Vec{2}((1.0, 1.0))))
        dh = DofHandler(grid); add!(dh, :u, ip^3); close!(dh)
        n_el   = ndofs_per_cell(dh)
        n_base = getnbasefunctions(ip)

        K    = allocate_matrix(dh)
        asmb = start_assemble(K, zeros(ndofs(dh)))
        ke   = zeros(n_el, n_el); re = zeros(n_el)
        for cell in CellIterator(dh)
            fill!(ke, 0.0); fill!(re, 0.0)
            reinit!(scv, cell)
            x = getcoordinates(cell)
            bending_tangent_KL!(ke, scv, zeros(n_el), mat)
            assemble!(asmb, celldofs(cell), ke, re)
        end

        # Project exact mode: u3_I = sin(πx_I)sin(πy_I), u1=u2=0
        u_h = zeros(ndofs(dh))
        for cell in CellIterator(dh)
            x  = getcoordinates(cell)
            cd = celldofs(cell)
            for I in 1:n_base
                u_h[cd[3I]] = sin(π * x[I][1]) * sin(π * x[I][2])
            end
        end

        0.5 * dot(u_h, K * u_h)
    end

    ws     = [kl_bending_energy_fem(n) for n in [2, 4, 8]]
    errors = abs.(ws .- W_exact)
    rates  = [log2(errors[i] / errors[i+1]) for i in 1:length(errors)-1]
    @test all(r -> r >= 1.5, rates)
    @test errors[end] / W_exact < 0.05
end
