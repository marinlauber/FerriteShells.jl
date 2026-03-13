using Documenter, FerriteShells

makedocs(
    modules = [FerriteShells],
    sitename = "FerriteShells.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", nothing) == "true",
        canonical="https://FerriteShells.github.io/FerriteShells.jl/",
        assets=String[],
        mathengine = MathJax3()
    ),
    authors = "Marin Lauber",
    pages = Any[
        "Introduction"      => "index.md",
        "Formulations"      => "formulations.md",
        "API reference"     => "reference/index.md",
    ]
)

# deploydocs(
#     repo = "github.com/marinlauber/FerriteShells.jl.git",
#     target = "build",
#     branch = "gh-pages",
#     push_preview = true,
#     versions = ["stable" => "v^", "v#.#" ],
# )