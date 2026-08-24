using FESolvers, SparseArrays, WriteVTK

abstract type AbstractShellProblem end
abstract type ReissnerMindlin end
abstract type KirchhoffLove end

mutable struct ShellProblem{S,T} <: AbstractShellProblem
    dh :: DofHandler
    ch :: ConstraintHandler
    scv :: ShellCellValues
    u :: Vector{T}
    r :: Vector{T}
    K :: SparseMatrixCSC{T, Int}
    mat :: LinearElastic
    pvd
end

function ReissnerMindlinShellProblem(dh, ch, scv, mat; pvd=nothing)
    K = allocate_matrix(dh)
    r = zeros(ndofs(dh))
    u = zeros(ndofs(dh))
    ShellProblem{ReissnerMindlin, eltype(u)}(dh, ch, scv, u, r, K, mat, pvd)
end
KirchhoffLoveShellProblem(args...) = error("Not implemented yet")

# pointers
FESolvers.getunknowns(p::AbstractShellProblem) = p.u
FESolvers.getresidual(p::AbstractShellProblem) = p.r
FESolvers.getjacobian(p::AbstractShellProblem) = p.K

# Update boundary conditions etc. for a new time step
function FESolvers.update_to_next_step!(p::AbstractShellProblem, t)
    update!(p.ch, t)     # Update Dirichlet boundary conditions
    apply!(FESolvers.getunknowns(p), p.ch)
end

function assemble_shell!(p::ShellProblem{S,T}, Δu) where {S<:ReissnerMindlin,T}
    Δu = isnothing(Δu) ? zero(p.u) : Δu
    n_e = ndofs_per_cell(p.dh)
    ke_i = zeros(n_e, n_e); re_i = zeros(n_e)
    asm_i = start_assemble(p.K, p.r)
    for cell in CellIterator(p.dh)
        fill!(ke_i, 0.0); fill!(re_i, 0.0)
        reinit!(p.scv, cell)
        sd  = shelldofs(cell)
        u_e = Δu[sd]
        membrane_residuals_RM!(re_i, p.scv, u_e, p.mat)
        bending_residuals_RM!(re_i, p.scv, u_e, p.mat)
        membrane_tangent_RM!(ke_i, p.scv, u_e, p.mat)
        bending_tangent_RM!(ke_i, p.scv, u_e, p.mat)
        assemble!(asm_i, sd, ke_i, re_i)
    end
    println("Done assembling shell problem")
end
assemble_shell!(p::ShellProblem{S,T}, Δu) where {S<:KirchhoffLove,T} = nothing

# Assemble stiffness and residual for x+=Δx
function FESolvers.update_problem!(p::AbstractShellProblem, Δu, update_spec)
    if !isnothing(Δu)
        apply_zero!(Δu, p.ch)
        p.u .+= Δu
    end
    # for linear problem, we can save some computations by only updating once per time step
    if FESolvers.should_update_jacobian(update_spec) || FESolvers.should_update_residual(update_spec)
        assemble_shell!(p, Δu) # dispatch to correct formulation
        apply_zero!(FESolvers.getjacobian(p), FESolvers.getresidual(p), p.ch)
    end
    println("Done updating problem")
end

# Get a scalar value to compare with the iteration tolerance
FESolvers.calculate_convergence_measure(p::AbstractShellProblem; tol=1e-6) = (println("Convergence measure: ", norm(p.r)); norm(p.r) < tol)

# Do all postprocessing for current step (after convergence)
function FESolvers.postprocess!(p::AbstractShellProblem, solver)
    step = FESolvers.get_step(solver)
    fname = isnothing(p.pvd) ? "shell" : first(split(p.pvd.path, "."))*"-i$step"
    VTKGridFile(fname, p.dh) do vtk
        write_solution(vtk, p.dh, p.u)
        !isnothing(p.pvd) && (p.pvd[step] = vtk)
    end
end

# Do stuff if required after the current time step has converged.
FESolvers.handle_converged!(p::AbstractShellProblem) = nothing

# Close any open file streams etc. Called in a `finally` block.
FESolvers.close_problem(p::AbstractShellProblem) = isnothing(p.pvd) ? nothing : vtk_save(p.pvd)