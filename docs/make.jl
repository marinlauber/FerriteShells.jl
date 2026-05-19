using Documenter, DocumenterCitations
using FerriteShells

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"), style=:numeric)

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
        "API reference"     => "reference/index.md",
    ],
    plugins=[bib]
)

deploydocs(
    repo = "github.com/marinlauber/FerriteShells.jl.git",
    target = "build",
    branch = "gh-pages",
    push_preview = true,
    versions = ["stable" => "v^", "v#.#" ],
)