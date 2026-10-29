```@meta
CurrentModule = FerriteShells
DocTestSetup = :(using FerriteShells)
```

# Shell Cell Values

## Main types

```@docs
ShellCellValues
NodeFrames
```

## Applicable functions
The following functions are applicable
`ShellCellValues`

```@docs
reinit!
function_value(::ShellCellValues, qp::Int, u_e::AbstractVector)
copy(::ShellCellValues)
```