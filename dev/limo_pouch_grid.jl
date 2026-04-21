using FerriteGmsh, Gmsh, FerriteShells

const W_POUCH = 0.10117598  # full width [m]
const H_POUCH = 0.109       # height [m]
const R_POUCH = 0.01388660  # top corner arc radius [m]

const Y_POUCH_LO = 0.02419037  # pouch region bottom y [m]
const Y_POUCH_HI = 0.10205670  # pouch region top y [m]
const X_INNER    = 0.01413512  # left/mid and mid/right x boundary [m]
const X_OUTER    = 0.04364469  # background/pouch x boundary [m]

"""
    make_limo_pouch_grid(nx, ny; order=2) -> Grid{3}

Structured transfinite mesh of the limo cross-section with pouch cell sets.
Outer geometry: rectangle (W_POUCH × H_POUCH) with quarter-circle arcs at the
top corners (radius R_POUCH). The mesh uses the same 3-patch decomposition as
`make_limo_grid` but scaled to the larger limo dimensions.

Boundary sets:
  "edge"        — straight bottom seam (y=0), nodes to be morphed
  "sym"         — outer walls + top arcs + top flat (symmetry / clamped)

Cell sets (assigned by element centroid):
  "background"  — all elements outside the pouch region
  "left_pouch"  — x ∈ [-X_OUTER, -X_INNER], y ∈ [Y_POUCH_LO, Y_POUCH_HI]
  "mid_pouch"   — x ∈ [-X_INNER,  X_INNER],  y ∈ [Y_POUCH_LO, Y_POUCH_HI]
  "right_pouch" — x ∈ [ X_INNER,  X_OUTER],  y ∈ [Y_POUCH_LO, Y_POUCH_HI]
"""
function make_limo_pouch_grid(nx, ny; order=2)
    Gmsh.initialize()
    gmsh.model.add("limo_pouch")

    p_bl = gmsh.model.geo.addPoint(-W_POUCH/2,           0.0,           0.0)
    p1   = gmsh.model.geo.addPoint(-W_POUCH/2 + R_POUCH, 0.0,           0.0)
    p2   = gmsh.model.geo.addPoint( W_POUCH/2 - R_POUCH, 0.0,           0.0)
    p_br = gmsh.model.geo.addPoint( W_POUCH/2,           0.0,           0.0)
    p3   = gmsh.model.geo.addPoint(-W_POUCH/2,           H_POUCH-R_POUCH, 0.0)
    p4   = gmsh.model.geo.addPoint( W_POUCH/2,           H_POUCH-R_POUCH, 0.0)
    pc1  = gmsh.model.geo.addPoint(-W_POUCH/2 + R_POUCH, H_POUCH-R_POUCH, 0.0)
    pc2  = gmsh.model.geo.addPoint( W_POUCH/2 - R_POUCH, H_POUCH-R_POUCH, 0.0)
    p7   = gmsh.model.geo.addPoint(-W_POUCH/2 + R_POUCH, H_POUCH,       0.0)
    p8   = gmsh.model.geo.addPoint( W_POUCH/2 - R_POUCH, H_POUCH,       0.0)

    c_bot_l = gmsh.model.geo.addLine(p_bl, p1)
    c_bot_m = gmsh.model.geo.addLine(p1,   p2)
    c_bot_r = gmsh.model.geo.addLine(p2,   p_br)
    c_left  = gmsh.model.geo.addLine(p3,   p_bl)
    c_right = gmsh.model.geo.addLine(p_br, p4)
    arc_tl  = gmsh.model.geo.addCircleArc(p7, pc1, p3)
    arc_tr  = gmsh.model.geo.addCircleArc(p4, pc2, p8)
    c_top_m = gmsh.model.geo.addLine(p7, p8)
    c_inn_l = gmsh.model.geo.addLine(p1, p7)
    c_inn_r = gmsh.model.geo.addLine(p2, p8)

    loop_l = gmsh.model.geo.addCurveLoop([c_bot_l, c_inn_l, arc_tl, c_left])
    surf_l = gmsh.model.geo.addPlaneSurface([loop_l])
    loop_m = gmsh.model.geo.addCurveLoop([c_bot_m, c_inn_r, -c_top_m, -c_inn_l])
    surf_m = gmsh.model.geo.addPlaneSurface([loop_m])
    loop_r = gmsh.model.geo.addCurveLoop([c_bot_r, c_right, arc_tr, -c_inn_r])
    surf_r = gmsh.model.geo.addPlaneSurface([loop_r])

    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(1, [c_bot_l, c_bot_m, c_bot_r],                -1, "edge")
    gmsh.model.addPhysicalGroup(1, [c_left, arc_tl, c_top_m, arc_tr, c_right], -1, "sym")
    gmsh.model.addPhysicalGroup(2, [surf_l, surf_m, surf_r],                    -1, "shell")

    n_arc = max(2, round(Int, nx * (π * R_POUCH / 2) / (W_POUCH - 2R_POUCH)))

    gmsh.model.mesh.setTransfiniteCurve(c_bot_m, nx + 1)
    gmsh.model.mesh.setTransfiniteCurve(c_top_m, nx + 1)
    gmsh.model.mesh.setTransfiniteCurve(c_inn_l, ny + 1)
    gmsh.model.mesh.setTransfiniteCurve(c_inn_r, ny + 1)
    gmsh.model.mesh.setTransfiniteCurve(c_bot_l, n_arc + 1)
    gmsh.model.mesh.setTransfiniteCurve(arc_tl,  n_arc + 1)
    gmsh.model.mesh.setTransfiniteCurve(c_left,  ny + 1)
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

    for setname in ("edge", "sym")
        ns = Set{Int}()
        for fi in getfacetset(grid, setname)
            for nid in Ferrite.facets(grid.cells[fi[1]])[fi[2]]
                push!(ns, nid)
            end
        end
        addnodeset!(grid, setname, ns)
    end

    bg_cells    = Set{Int}()
    left_cells  = Set{Int}()
    mid_cells   = Set{Int}()
    right_cells = Set{Int}()

    for (ci, cell) in enumerate(grid.cells)
        cx = sum(grid.nodes[n].x[1] for n in cell.nodes) / length(cell.nodes)
        cy = sum(grid.nodes[n].x[2] for n in cell.nodes) / length(cell.nodes)
        if Y_POUCH_LO ≤ cy ≤ Y_POUCH_HI && abs(cx) ≤ X_OUTER
            if     cx ≤ -X_INNER  push!(left_cells,  ci)
            elseif cx ≥  X_INNER  push!(right_cells, ci)
            else                   push!(mid_cells,   ci)
            end
        else
            push!(bg_cells, ci)
        end
    end

    addcellset!(grid, "background",  bg_cells)
    addcellset!(grid, "left_pouch",  left_cells)
    addcellset!(grid, "mid_pouch",   mid_cells)
    addcellset!(grid, "right_pouch", right_cells)

    return grid
end
