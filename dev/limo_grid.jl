using FerriteGmsh, Gmsh, FerriteShells

const W = 0.10117598   # full width [m]
const H = 0.109        # height [m]
const R = 0.012740     # top corner radius [m]

"""
    make_limo_grid(nx, ny; order=2) -> Grid{3}

Structured transfinite Q9 mesh of the flat limo cross-section: a rectangle
(W × H) with two quarter-circle arcs of radius R at the TOP corners.
Geometry and patch decomposition:

    p3 arc  p7 ──────────── p8  arc p4
    │  left │     middle     │ right │
    p_bl── p1 ──────────── p2 ────p_br
           └──────────────────┘
                  "edge"

Boundary sets added to the returned grid:
  "edge" — straight bottom seam (y=0), nodes to be morphed
  "sym"  — left wall + top arcs + top flat + right wall (u_z=0, θ=0)
"""
function make_limo_grid(nx, ny; order=2)
    Gmsh.initialize()
    gmsh.model.add("limo")

    # Bottom corners and inner patch boundaries
    p_bl = gmsh.model.geo.addPoint(-W/2,      0.0,   0.0)
    p1   = gmsh.model.geo.addPoint(-W/2 + R,  0.0,   0.0)
    p2   = gmsh.model.geo.addPoint( W/2 - R,  0.0,   0.0)
    p_br = gmsh.model.geo.addPoint( W/2,      0.0,   0.0)
    # Arc endpoints on vertical walls at height H-R
    p3   = gmsh.model.geo.addPoint(-W/2,       H - R, 0.0)
    p4   = gmsh.model.geo.addPoint( W/2,       H - R, 0.0)
    # Arc centres
    pc1  = gmsh.model.geo.addPoint(-W/2 + R,  H - R, 0.0)
    pc2  = gmsh.model.geo.addPoint( W/2 - R,  H - R, 0.0)
    # Top inner patch boundaries
    p7   = gmsh.model.geo.addPoint(-W/2 + R,  H,     0.0)
    p8   = gmsh.model.geo.addPoint( W/2 - R,  H,     0.0)

    # Boundary curves
    c_bot_l = gmsh.model.geo.addLine(p_bl, p1)           # bottom-left segment
    c_bot_m = gmsh.model.geo.addLine(p1,   p2)           # bottom-middle segment
    c_bot_r = gmsh.model.geo.addLine(p2,   p_br)         # bottom-right segment
    c_left  = gmsh.model.geo.addLine(p3,   p_bl)         # left outer wall (downward)
    c_right = gmsh.model.geo.addLine(p_br, p4)           # right outer wall (upward)
    arc_tl  = gmsh.model.geo.addCircleArc(p7, pc1, p3)  # top-left arc: p7→p3
    arc_tr  = gmsh.model.geo.addCircleArc(p4, pc2, p8)  # top-right arc: p4→p8
    c_top_m = gmsh.model.geo.addLine(p7, p8)             # top flat middle
    # Internal patch boundaries
    c_inn_l = gmsh.model.geo.addLine(p1,  p7)            # inner left vertical
    c_inn_r = gmsh.model.geo.addLine(p2,  p8)            # inner right vertical

    # CCW curve loops
    # Left strip:   p_bl → p1 → p7 → p3 → p_bl
    loop_l = gmsh.model.geo.addCurveLoop([c_bot_l, c_inn_l, arc_tl, c_left])
    surf_l = gmsh.model.geo.addPlaneSurface([loop_l])
    # Middle rect:  p1 → p2 → p8 → p7 → p1
    loop_m = gmsh.model.geo.addCurveLoop([c_bot_m, c_inn_r, -c_top_m, -c_inn_l])
    surf_m = gmsh.model.geo.addPlaneSurface([loop_m])
    # Right strip:  p2 → p_br → p4 → p8 → p2
    loop_r = gmsh.model.geo.addCurveLoop([c_bot_r, c_right, arc_tr, -c_inn_r])
    surf_r = gmsh.model.geo.addPlaneSurface([loop_r])

    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(1, [c_bot_l, c_bot_m, c_bot_r],                   -1, "edge")
    gmsh.model.addPhysicalGroup(1, [c_left, arc_tl, c_top_m, arc_tr, c_right],    -1, "sym")
    gmsh.model.addPhysicalGroup(2, [surf_l, surf_m, surf_r],                       -1, "shell")

    # Arc node count proportional to arc length / flat-bottom element size
    n_arc = max(2, round(Int, nx * (π * R / 2) / (W - 2R)))

    # Middle patch
    gmsh.model.mesh.setTransfiniteCurve(c_bot_m, nx + 1)
    gmsh.model.mesh.setTransfiniteCurve(c_top_m, nx + 1)
    gmsh.model.mesh.setTransfiniteCurve(c_inn_l, ny + 1)
    gmsh.model.mesh.setTransfiniteCurve(c_inn_r, ny + 1)
    # Left strip (n_arc horizontal, ny vertical — must match c_inn_l)
    gmsh.model.mesh.setTransfiniteCurve(c_bot_l, n_arc + 1)
    gmsh.model.mesh.setTransfiniteCurve(arc_tl,  n_arc + 1)
    gmsh.model.mesh.setTransfiniteCurve(c_left,  ny + 1)
    # Right strip
    gmsh.model.mesh.setTransfiniteCurve(c_bot_r, n_arc + 1)
    gmsh.model.mesh.setTransfiniteCurve(arc_tr,  n_arc + 1)
    gmsh.model.mesh.setTransfiniteCurve(c_right, ny + 1)

    gmsh.model.mesh.setTransfiniteSurface(surf_l, "Left", [p_bl, p1, p7, p3])
    gmsh.model.mesh.setTransfiniteSurface(surf_m, "Left", [p1,   p2, p8, p7])
    gmsh.model.mesh.setTransfiniteSurface(surf_r, "Left", [p2, p_br, p4, p8])
    gmsh.model.mesh.setRecombine(2, surf_l)
    gmsh.model.mesh.setRecombine(2, surf_m)
    gmsh.model.mesh.setRecombine(2, surf_r)

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)

    grid = togrid()
    Gmsh.finalize()

    grid = shell_grid(grid)

    # Derive nodesets from the physical-group facetsets
    for setname in ("edge", "sym")
        nodes = Set{Int}()
        for fi in getfacetset(grid, setname)
            for nid in Ferrite.facets(grid.cells[fi[1]])[fi[2]]
                push!(nodes, nid)
            end
        end
        addnodeset!(grid, setname, nodes)
    end

    return grid
end
