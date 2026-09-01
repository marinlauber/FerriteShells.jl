using Documenter, DocumenterCitations, DocumenterCodeBlocks
using FerriteShells

# run Literate on the code gallery sources to generate the markdown pages
include("generate.jl")

bibtex_plugin = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"), style=:numeric)

codeblocks_plugin = CodeBlocks(line_counter=:named)

makedocs(
    modules = [FerriteShells],
    sitename = "FerriteShells.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", nothing) == "true",
        canonical="https://marinlauber.github.io/FerriteShells.jl/",
        assets=String[],
        mathengine = mathengine = MathJax3(Dict(
            :loader => Dict("load" => ["[tex]/physics"]),
            :tex => Dict(
                "inlineMath" => [["\$","\$"], ["\\(","\\)"]],
                "tags" => "ams",
                "packages" => ["base", "ams", "autoload", "physics"])
            )
        )
    ),
    authors = "Marin Lauber",
    pages = Any[
        "Introduction"      => "index.md",
        "Formulations"      => ["shell.md", "KirchhoffLove.md",
                                "ReissnerMindlin.md", "shell_models.md",
                                "solvers.md", "References.md"],
        "Code gallery" => [
            "Code gallery overview" => "gallery/index.md",
            "gallery/linear-elasticity.md",
        ],
        "API reference"     => "reference/index.md",
    ],
    plugins=[
        bibtex_plugin,
        codeblocks_plugin,
    ]
)

deploydocs(
    repo = "github.com/marinlauber/FerriteShells.jl.git",
    target = "build",
    branch = "gh-pages",
    push_preview = true,
    versions = ["stable" => "v^", "v#.#" ],
)