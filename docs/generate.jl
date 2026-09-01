# Generate the code gallery pages from the Literate sources.
import Literate

# Code gallery: `.jl` sources live in `literate-gallery`, the rendered pages are
# written to `gallery` (generated, not tracked by git).
GALLERY_IN = joinpath(@__DIR__, "src", "literate-gallery")
GALLERY_OUT = joinpath(@__DIR__, "src", "gallery")
mkpath(GALLERY_OUT)

# Skip execution of these files, e.g. because they need dependencies that are
# not available in the docs environment.
DONT_EXECUTE = Set{String}()

for program in readdir(GALLERY_IN; join = true)
    name = basename(program)
    if endswith(program, ".jl")
        skip_execution = name ∈ DONT_EXECUTE

        # The comment-free script that is spliced into the "Plain program"
        # section through the `@__CODE__` placeholder.
        if skip_execution
            code_clean = "<< script output is skipped for this example >>"
        else
            script = Literate.script(program, GALLERY_OUT)
            code = strip(read(script, String))
            line_ending = occursin("\r\n", code) ? "\r\n" : "\n"
            code_clean = join(filter(l -> !endswith(l, "#hide"), split(code, r"\n|\r\n")), line_ending)
            code_clean = replace(code_clean, r"^# This file was generated .*$"m => "")
            code_clean = strip(code_clean)
        end

        mdpost(str) = replace(str, "@__CODE__" => code_clean)

        if skip_execution
            # Plain code fence instead of the default `@example`, so that
            # Documenter does not run the code.
            Literate.markdown(program, GALLERY_OUT; postprocess = mdpost, codefence = "````julia" => "````")
        else
            Literate.markdown(program, GALLERY_OUT; postprocess = mdpost)
        end
    elseif any(endswith(program, ext) for ext in (".png", ".jpg", ".gif", ".webp"))
        cp(program, joinpath(GALLERY_OUT, name); force = true)
    else
        @warn "ignoring $program"
    end
end

# Remove the VTK files the examples write while executing; they should not be
# deployed with the docs.
const VTK_EXTENSIONS = (".vtu", ".vtk", ".pvd", ".vtkhdf")
for file in readdir(GALLERY_OUT; join = true)
    any(endswith(file, ext) for ext in VTK_EXTENSIONS) && rm(file)
end

# Gallery overview page. The body is kept in `docs/` (outside `src/`) so that
# Documenter does not render it as a page of its own, and is copied into the
# generated `gallery` directory as `index.md`.
function write_overview(dir, body_file)
    io = IOBuffer()
    # Point Documenter's "Edit source" button at the tracked body file rather
    # than at the generated `index.md`, which does not exist in the repository.
    println(io, "```@meta")
    println(io, "EditURL = \"", replace(relpath(body_file, dir), '\\' => '/'), "\"")
    println(io, "```\n")
    write(io, read(body_file, String))
    index_md = joinpath(dir, "index.md")
    content = String(take!(io))
    # Only write when the content changed, to not retrigger LiveServer.
    if !isfile(index_md) || read(index_md, String) != content
        write(index_md, content)
    end
end

write_overview(GALLERY_OUT, joinpath(@__DIR__, "gallery_index_body.md"))
