# Assembly cost: explicit index-notation kernels vs. ForwardDiff (FD) of `energy_RM`.
#
# Two figures for reporting/presentation:
#   1. assembly_kernel_cost.png — per-element residual+tangent time and allocations,
#      for every element family, with and without MITC shear tying.
#   2. assembly_scaling.png     — global assembly of a flat plate vs. mesh size.
#
# Both paths compute the *same* residual and tangent (checked to machine precision
# below); only the derivative route differs:
#   explicit : membrane_/bending_residuals_RM! + membrane_/bending_tangent_RM!
#   FD       : residuals_RM_FD! (ForwardDiff.gradient) + tangent_RM_FD! (…hessian)
#
#   julia --project=. AssemblyBenchmark.jl            # measure, then plot
#   julia --project=. AssemblyBenchmark.jl --replot   # re-plot cached measurements
#
# Measuring takes ~2 min; results are cached next to this file so the figures can
# be restyled without re-running the benchmarks.

using FerriteShells, BenchmarkTools, CairoMakie, LinearAlgebra, Printf, Serialization

const IMG_DIR   = joinpath(@__DIR__, "..", "docs", "src", "images")
const CACHEFILE = joinpath(@__DIR__, "assembly_benchmark.jls")

# Palette: categorical slots 1 (blue) and 2 (orange) of the validated default
# data-viz palette, used unchanged; chrome/ink tokens from the same set.
const C_EXPLICIT = "#2a78d6"
const C_FD       = "#eb6834"
const SURFACE    = "#fcfcfb"
const INK        = "#0b0b0b"
const INK2       = "#52514e"
const MUTED      = "#898781"
const GRIDCOLOR  = "#e1e0d9"

const MAT = LinearElastic(1.0e6, 0.3, 0.1)

# (primitive, refshape, order, MITC variant, element label, tying label)
const CONFIGS = [
    (Triangle,               RefTriangle,      1, MITC3,  "Tri P1",  "MITC3"),
    (Quadrilateral,          RefQuadrilateral, 1, MITC4,  "Quad P1", "MITC4"),
    (QuadraticTriangle,      RefTriangle,      2, MITC6a, "Tri P2",  "MITC6a"),
    (QuadraticQuadrilateral, RefQuadrilateral, 2, MITC9,  "Quad P2", "MITC9"),
]

# ---------------------------------------------------------------------------
# per-element kernels
# ---------------------------------------------------------------------------

function explicit_kernel!(ke, re, scv, u_e, mat)
    fill!(ke, 0.0); fill!(re, 0.0)
    membrane_residuals_RM!(re, scv, u_e, mat)
    bending_residuals_RM!(re, scv, u_e, mat)
    membrane_tangent_RM!(ke, scv, u_e, mat)
    bending_tangent_RM!(ke, scv, u_e, mat)
    return nothing
end

function fd_kernel!(ke, re, scv, u_e, mat)
    fill!(ke, 0.0); fill!(re, 0.0)
    residuals_RM_FD!(re, scv, u_e, mat)
    tangent_RM_FD!(ke, scv, u_e, mat)
    return nothing
end

# Time (seconds) and allocations (bytes) of one residual+tangent evaluation.
function kernel_cost(kernel!, scv, u_e, ke, re, mat)
    kernel!(ke, re, scv, u_e, mat)                       # warm up / compile
    t = @belapsed $kernel!($ke, $re, $scv, $u_e, $MAT) evals=1
    a = @allocated kernel!(ke, re, scv, u_e, mat)
    return t, a
end

function kernel_benchmarks()
    rows = NamedTuple[]
    for (primitive, refshape, order, mitc, label, tying) in CONFIGS
        grid = shell_grid(generate_grid(primitive, (2, 2)))
        ip   = Lagrange{refshape, order}()
        qr   = QuadratureRule{refshape}(order + 1)
        dh   = DofHandler(grid)
        add!(dh, :u, ip^3); add!(dh, :θ, ip^2); close!(dh)

        scv_mitc = ShellCellValues(qr, ip, ip; mitc = mitc)
        scv_no   = ShellCellValues(qr, ip, ip)
        cell     = first(CellIterator(dh))
        reinit!(scv_mitc, cell); reinit!(scv_no, cell)

        n_e = ndofs_per_cell(dh)
        u_e = 0.001 .* randn(n_e)
        ke  = zeros(n_e, n_e); re = zeros(n_e)
        ke2 = zeros(n_e, n_e); re2 = zeros(n_e)

        for (shear, scv) in (("no MITC", scv_no), ("MITC", scv_mitc))
            # the two routes must agree — otherwise the timing comparison is meaningless
            explicit_kernel!(ke, re, scv, u_e, MAT)
            fd_kernel!(ke2, re2, scv, u_e, MAT)
            dr = norm(re - re2) / norm(re2)
            dk = norm(ke - ke2) / norm(ke2)
            @assert dr < 1e-10 && dk < 1e-10 "explicit and FD disagree ($label, $shear)"

            t_exp, a_exp = kernel_cost(explicit_kernel!, scv, u_e, ke,  re,  MAT)
            t_fd,  a_fd  = kernel_cost(fd_kernel!,       scv, u_e, ke2, re2, MAT)
            push!(rows, (; label, tying, shear, n_e, t_exp, a_exp, t_fd, a_fd, dr, dk))
            @printf("  %-8s %-8s n_e=%2d  explicit %8.2f µs %7d B | FD %9.2f µs %7d B | %5.1f×\n",
                    label, shear, n_e, t_exp * 1e6, a_exp, t_fd * 1e6, a_fd, t_fd / t_exp)
        end
    end
    return rows
end

# ---------------------------------------------------------------------------
# global assembly on a flat plate
# ---------------------------------------------------------------------------

# `shelldofs` allocates its permutation vector once per cell; the benchmark uses a
# preallocated buffer so the measurement isolates the element-kernel cost.
function shelldofs!(perm, cell)
    dofs = cell.dofs
    n = length(dofs) ÷ 5
    for I in 1:n
        @views perm[5I-4:5I-2] .= dofs[3I-2:3I]
        perm[5I-1] = dofs[3n + 2I-1]
        perm[5I  ] = dofs[3n + 2I]
    end
    return perm
end

function assemble_global!(kernel!, K, r, dh, scv, u, mat, ke, re, u_e, perm)
    assembler = start_assemble(K, r)
    for cell in CellIterator(dh)
        reinit!(scv, cell)
        shelldofs!(perm, cell)
        for (i, d) in enumerate(perm)
            u_e[i] = u[d]
        end
        kernel!(ke, re, scv, u_e, mat)
        assemble!(assembler, perm, ke, re)
    end
    return nothing
end

function scaling_benchmarks(ns = (4, 8, 16, 32, 64); reps = 3)
    ip  = Lagrange{RefQuadrilateral, 1}()
    qr  = QuadratureRule{RefQuadrilateral}(2)
    rows = NamedTuple[]
    for n in ns
        grid = shell_grid(generate_grid(Quadrilateral, (n, n)))
        dh   = DofHandler(grid)
        add!(dh, :u, ip^3); add!(dh, :θ, ip^2); close!(dh)
        K = allocate_matrix(dh); r = zeros(ndofs(dh)); u = 0.001 .* randn(ndofs(dh))
        n_e  = ndofs_per_cell(dh)
        ke   = zeros(n_e, n_e); re = zeros(n_e)
        u_e  = zeros(n_e); perm = zeros(Int, n_e)
        ncells = getncells(grid)

        for (shear, scv) in (("no MITC", ShellCellValues(qr, ip, ip)),
                             ("MITC",    ShellCellValues(qr, ip, ip; mitc = MITC4)))
            for (method, kernel!) in (("explicit", explicit_kernel!), ("FD", fd_kernel!))
                run!() = assemble_global!(kernel!, K, r, dh, scv, u, MAT, ke, re, u_e, perm)
                run!()                                     # warm up / compile
                t = minimum(@elapsed(run!()) for _ in 1:reps)
                a = @allocated run!()
                push!(rows, (; ncells, shear, method, t, a))
                @printf("  %5d cells  %-8s %-8s  %8.2f ms  %10.3f MiB\n",
                        ncells, shear, method, t * 1e3, a / 2^20)
            end
        end
    end
    return rows
end

# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

pick(rows, shear, f) = [f(r) for r in rows if r.shear == shear]

fmt_time(t) = t * 1e6 < 1e3 ? @sprintf("%.1f", t * 1e6) : @sprintf("%.0f", t * 1e6)
fmt_bytes(a) = a == 0 ? "0 B" : @sprintf("%.0f kB", a / 1024)

function style!(ax)
    ax.backgroundcolor = SURFACE
    ax.xgridvisible = false
    ax.ygridcolor = GRIDCOLOR
    ax.ygridwidth = 1
    ax.topspinevisible = false
    ax.rightspinevisible = false
    ax.leftspinecolor = GRIDCOLOR
    ax.bottomspinecolor = GRIDCOLOR
    ax.xticklabelcolor = INK2
    ax.yticklabelcolor = MUTED
    ax.xtickcolor = GRIDCOLOR
    ax.ytickcolor = GRIDCOLOR
    ax.titlecolor = INK
    ax.xlabelcolor = INK2
    ax.ylabelcolor = INK2
    return ax
end

const BARW  = 0.36   # bar width, leaving a surface gap between the pair
const OFFS  = 0.19   # centre offset of each bar in its slot

function kernel_figure(rows; filename = "assembly_kernel_cost.png")
    # left column has no tying, so only the right column names the MITC variant
    plain_labels = [c[5] for c in CONFIGS]
    mitc_labels  = [c[5] * "\n" * c[6] for c in CONFIGS]
    xs = 1:length(CONFIGS)

    fig = Figure(size = (1020, 720), backgroundcolor = SURFACE, figure_padding = 18)
    Label(fig[0, 1:2], "Per-element residual + tangent: explicit kernels vs. ForwardDiff",
          fontsize = 19, color = INK, font = :bold, halign = :left, padding = (4, 0, 0, 0))
    Label(fig[1, 1:2], "Reissner–Mindlin shell element, LinearElastic material — same result to machine precision",
          fontsize = 13, color = INK2, halign = :left, padding = (4, 0, 0, 6))

    for (col, shear) in enumerate(("no MITC", "MITC"))
        labels = col == 1 ? plain_labels : mitc_labels
        title  = col == 1 ? "no shear tying" : "with MITC shear tying"
        t_exp = pick(rows, shear, r -> r.t_exp * 1e6)
        t_fd  = pick(rows, shear, r -> r.t_fd  * 1e6)
        a_exp = pick(rows, shear, r -> r.a_exp / 1024)
        a_fd  = pick(rows, shear, r -> r.a_fd  / 1024)

        # --- time (log scale) ---
        axt = Axis(fig[2, col]; title, titlesize = 15,
                   ylabel = col == 1 ? "time per element  [µs]" : "",
                   yscale = log10, xticks = (xs, labels), xticklabelsize = 11)
        style!(axt)
        barplot!(axt, xs .- OFFS, t_exp; width = BARW, color = C_EXPLICIT,
                 fillto = 1.0, label = "explicit")
        barplot!(axt, xs .+ OFFS, t_fd;  width = BARW, color = C_FD,
                 fillto = 1.0, label = "ForwardDiff")
        text!(axt, xs .- OFFS, t_exp; text = fmt_time.(t_exp ./ 1e6), align = (:center, :bottom),
              offset = (0, 4), fontsize = 11, color = INK2)
        text!(axt, xs .+ OFFS, t_fd;  text = fmt_time.(t_fd ./ 1e6), align = (:center, :bottom),
              offset = (0, 4), fontsize = 11, color = INK2)
        # speed-up annotation between the pair
        text!(axt, xs, t_fd .* 2.6; text = [@sprintf("%.0f×", f / e) for (e, f) in zip(t_exp, t_fd)],
              align = (:center, :bottom), fontsize = 12, color = INK, font = :bold)
        ylims!(axt, 1.0, maximum(t_fd) * 14)

        # --- allocations (linear, kB) ---
        axa = Axis(fig[3, col]; ylabel = col == 1 ? "allocations per element  [kB]" : "",
                   xticks = (xs, labels), xticklabelsize = 11)
        style!(axa)
        barplot!(axa, xs .- OFFS, a_exp; width = BARW, color = C_EXPLICIT)
        barplot!(axa, xs .+ OFFS, a_fd;  width = BARW, color = C_FD)
        text!(axa, xs .- OFFS, a_exp; text = fmt_bytes.(pick(rows, shear, r -> r.a_exp)),
              align = (:center, :bottom), offset = (0, 4), fontsize = 11, color = INK2)
        text!(axa, xs .+ OFFS, a_fd;  text = fmt_bytes.(pick(rows, shear, r -> r.a_fd)),
              align = (:center, :bottom), offset = (0, 4), fontsize = 11, color = INK2)
        ylims!(axa, 0, maximum(a_fd) * 1.25)
    end

    linkyaxes!(contents(fig[2, 1])[1], contents(fig[2, 2])[1])
    linkyaxes!(contents(fig[3, 1])[1], contents(fig[3, 2])[1])

    Legend(fig[4, 1:2], [PolyElement(color = C_EXPLICIT), PolyElement(color = C_FD)],
           ["explicit index-notation kernels", "ForwardDiff of energy_RM"],
           orientation = :horizontal, framevisible = false, labelcolor = INK2,
           labelsize = 13, padding = (0, 0, 0, 0))
    rowgap!(fig.layout, 8)
    save(joinpath(IMG_DIR, filename), fig)
    return fig
end

function scaling_figure(rows; filename = "assembly_scaling.png")
    ncells = sort(unique(r.ncells for r in rows))
    series = (("explicit", "MITC",    C_EXPLICIT, :solid,  "explicit, MITC"),
              ("explicit", "no MITC", C_EXPLICIT, :dash,   "explicit, no MITC"),
              ("FD",       "MITC",    C_FD,       :solid,  "ForwardDiff, MITC"),
              ("FD",       "no MITC", C_FD,       :dash,   "ForwardDiff, no MITC"))
    get(method, shear, f) = [f(r) for n in ncells
                             for r in rows if r.ncells == n && r.method == method && r.shear == shear]

    fig = Figure(size = (1020, 430), backgroundcolor = SURFACE, figure_padding = 18)
    Label(fig[0, 1:2], "Global assembly of a flat plate — wall time and total allocations",
          fontsize = 19, color = INK, font = :bold, halign = :left, padding = (4, 0, 0, 0))
    Label(fig[1, 1:2], "Quadrilateral P1 shell elements, tangent + residual, one full sweep over the mesh",
          fontsize = 13, color = INK2, halign = :left, padding = (4, 0, 0, 6))

    axt = Axis(fig[2, 1]; xlabel = "number of elements", ylabel = "assembly time  [ms]",
               xscale = log10, yscale = log10,
               xticks = (ncells, string.(ncells)), title = "time", titlesize = 15)
    axa = Axis(fig[2, 2]; xlabel = "number of elements", ylabel = "allocated  [KiB]",
               xscale = log10, yscale = log10, xticks = (ncells, string.(ncells)),
               title = "memory", titlesize = 15)
    style!(axt); style!(axa)

    for (method, shear, color, ls, label) in series
        t = get(method, shear, r -> r.t * 1e3)
        a = get(method, shear, r -> r.a / 1024)
        scatterlines!(axt, ncells, t; color, linestyle = ls, linewidth = 2,
                      markersize = 9, strokecolor = SURFACE, strokewidth = 1.5, label)
        scatterlines!(axa, ncells, a; color, linestyle = ls, linewidth = 2,
                      markersize = 9, strokecolor = SURFACE, strokewidth = 1.5, label)
    end

    # Direct labels: the time curves separate, so label each; the memory curves sit on
    # top of each other per method, so label once per method.
    for (method, shear, color, ls, label) in series
        t = get(method, shear, r -> r.t * 1e3)
        text!(axt, last(ncells), last(t); text = " " * label, align = (:left, :center),
              fontsize = 11, color = INK2)
    end
    a_fd  = get("FD",       "MITC", r -> r.a / 1024)
    a_exp = get("explicit", "MITC", r -> r.a / 1024)
    mid   = (length(ncells) + 1) ÷ 2
    text!(axa, ncells[mid], a_fd[mid] * 5.0;
          text = "ForwardDiff — grows with the mesh\n(MITC and no MITC overlap)",
          align = (:center, :bottom), fontsize = 12, color = INK2)
    text!(axa, ncells[mid], a_exp[mid] * 2.4;
          text = "explicit — one Assembler, nothing per element\n(MITC and no MITC overlap)",
          align = (:center, :bottom), fontsize = 12, color = INK2)
    xlims!(axt, first(ncells) / 1.6, last(ncells) * 9)
    xlims!(axa, first(ncells) / 2.2, last(ncells) * 2.2)
    ylims!(axa, 0.55, maximum(a_fd) * 12)

    Legend(fig[3, 1:2],
           [LineElement(color = C_EXPLICIT, linestyle = :solid, linewidth = 2),
            LineElement(color = C_EXPLICIT, linestyle = :dash,  linewidth = 2),
            LineElement(color = C_FD,       linestyle = :solid, linewidth = 2),
            LineElement(color = C_FD,       linestyle = :dash,  linewidth = 2)],
           ["explicit, MITC", "explicit, no MITC", "ForwardDiff, MITC", "ForwardDiff, no MITC"],
           orientation = :horizontal, framevisible = false, labelcolor = INK2, labelsize = 13)
    rowgap!(fig.layout, 8)
    save(joinpath(IMG_DIR, filename), fig)
    return fig
end

# ---------------------------------------------------------------------------

function markdown_table(rows)
    println("\n| Element | tying | shear | DOFs | explicit (µs) | FD (µs) | speed-up | explicit (B) | FD (B) |")
    println("|---|---|---|---|---|---|---|---|---|")
    for r in rows
        @printf("| %s | %s | %s | %d | %.1f | %.1f | %.0f× | %d | %d |\n",
                r.label, r.tying, r.shear, r.n_e,
                r.t_exp * 1e6, r.t_fd * 1e6, r.t_fd / r.t_exp, r.a_exp, r.a_fd)
    end
end

replot = "--replot" in ARGS && isfile(CACHEFILE)
if replot
    kernel_rows, scaling_rows = deserialize(CACHEFILE)
    println("re-plotting cached measurements from $(CACHEFILE)")
else
    println("Per-element kernels:")
    kernel_rows = kernel_benchmarks()
    println("\nGlobal assembly:")
    scaling_rows = scaling_benchmarks()
    serialize(CACHEFILE, (kernel_rows, scaling_rows))
end

kernel_figure(kernel_rows)
scaling_figure(scaling_rows)
markdown_table(kernel_rows)
println("\nfigures written to $(normpath(IMG_DIR))")
