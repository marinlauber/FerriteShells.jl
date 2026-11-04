# Reference for ../roof_d3.jl: FerriteShells RM, NodeFrames, no MITC (same averaged normals as the prototype).
# Run from the repository root with the package environment: julia --project=. dev/d3_prototype/reference/roof_ferriteshells.jl
using FerriteShells
const R, L, Φ = 25.0, 50.0, 40π/180
function roof(C, N)
    g = shell_grid(generate_grid(C, (N, N), Vec{2}((-Φ, 0.0)), Vec{2}((Φ, L)));
                   map = n -> (n.x[2], R*cos(n.x[1]), R*sin(n.x[1])))
    addnodeset!(g, "diaphragm", x -> x[1] ≈ 0.0 || x[1] ≈ L)
    addnodeset!(g, "ref", x -> abs(x[1]-L/2)<1e-8 && abs(x[2]-R*cos(Φ))<1e-8 && abs(x[3]-R*sin(Φ))<1e-8)
    g
end
function solve(C, ip, qr, mitc, N; nodeframes=false)
    grid = roof(C, N)
    frames = nodeframes ? NodeFrames(grid, ip) : nothing
    scv = ShellCellValues(qr, ip, ip; mitc, frames)
    mat = LinearElastic(4.32e8, 0.0, 0.25)
    dh = DofHandler(grid); add!(dh, :u, ip^3); add!(dh, :θ, ip^2); close!(dh)
    nb = getnbasefunctions(ip)
    K = allocate_matrix(dh); f = zeros(ndofs(dh))
    asmb = start_assemble(K, zeros(ndofs(dh)))
    ke = zeros(5nb,5nb); re = zeros(5nb); fe = zeros(5nb); u0 = zeros(5nb)
    q = Vec{3}((0.0, -90.0, 0.0))
    for cell in CellIterator(dh)
        fill!(ke,0); fill!(re,0); fill!(fe,0)
        reinit!(scv, cell)
        membrane_tangent_RM!(ke, scv, u0, mat); bending_tangent_RM!(ke, scv, u0, mat)
        sd = shelldofs(cell); assemble!(asmb, sd, ke, re)
        for qp in 1:getnquadpoints(scv), I in 1:nb
            @views fe[5I-4:5I-2] .+= scv.N[I,qp] * q * scv.detJdV[qp]
        end
        @views f[sd] .+= fe
    end
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getnodeset(grid,"diaphragm"), x->zeros(2), [2,3]))
    add!(ch, Dirichlet(:u, getnodeset(grid,"ref"), x->0.0, [1]))
    close!(ch); Ferrite.update!(ch, 0.0); apply!(K, f, ch)
    a = K \ f
    r = only(getnodeset(grid,"ref"))
    for cell in CellIterator(dh), (I,gid) in enumerate(getnodes(cell))
        gid == r && return -a[celldofs(cell)[3I-1]] / 0.3024, ndofs(dh)
    end
end
Q4 = Lagrange{RefQuadrilateral,1}(); Q9 = Lagrange{RefQuadrilateral,2}()
variants = [("Q4 NF",Quadrilateral,Q4,QuadratureRule{RefQuadrilateral}(2),nothing,true),("Q9 NF",QuadraticQuadrilateral,Q9,QuadratureRule{RefQuadrilateral}(3),nothing,true)]
nels = [4, 8, 16, 32]
for (name, C, ip, qr, m, nf) in variants
    try
        r = [solve(C, ip, qr, m, n; nodeframes=nf) for n in nels]
        solve(C, ip, qr, m, 32; nodeframes=nf)
        t = @elapsed solve(C, ip, qr, m, 32; nodeframes=nf)
        println(rpad(name,12), join(map(x->lpad(string(round(x[1];digits=10)),14), r)), "   n=32 ndofs=", r[end][2], " t=", round(t,digits=3), " s")
    catch e
        println(rpad(name,12), "ERROR ", sprint(showerror, e)[1:min(end,200)])
    end
end
