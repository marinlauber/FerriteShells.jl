# Code style

- Do not use long separator comments like `# ------` or `# ======` in code.
- In Julia:
    - Use implicit return where possible
    - Use `@views` when slicing arrays to avoid unnecessary allocations.
    - Use `@inbounds` to skip bounds checking in performance-critical code, but only when you are sure it is safe to do so.
    - Use @inline
    - prefer function argument on single line, rather than having them on multiple lines, unless the function signature is very long, then use 2-3 lines.
- In Julia tests, prefer a single `@testset` with multiple `@test` statements over many separate nested `@testset` blocks with one `@test` each.