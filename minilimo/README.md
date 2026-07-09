## miniLIMO setups

1. `limo_dynamic_coupled.jl`: works, dynamic morphing (damped) and then static Lie-Trotter coupling (weak) with the WK3 model.Pact max reaches 400 mmHg.
2. `limo_dynamic_kostas_single.jl`: static inflation for Kostas' project. Runs well via command line Pact arguments
3. `limo_dynamic_coupled_transient.jl`: dynamic morphing (damped) and then dynamic Lie-Trotter coupling (weak) with the WK3 model.
4. `limo_dynamic_coupled_strong.jl`:  dynamic morphing (damped) and then dynamic, stronlgy coupled Lie-Trotter coupling with the WK3 model.
5. `limo_dynamic_full.jl` same as the `1`, but with the full geometry, no symmetry plane.
6. `utils`: store some repeated functions that are used across these files

## old files

```julia
limo_dynamic_actuation.jl
limo_inflation.jl
limo_morph_bypass.jl
limo_ptc_inflation.jl
limo_dynamic_full.jl
limo_dynamic.jl
```