

function element_membrane_residual!(re, scv, u_e, material)

    fill!(re, 0.0)

    nqp = getnquadpoints(scv)

    for q in 1:nqp

        # --- Geometry ---
        geom = update!(scv, q, u_e)

        a1 = geom.a1
        a2 = geom.a2

        A_metric = geom.A_metric
        a_metric = geom.a_metric

        # --- Strain ---
        E = 0.5 * (a_metric - A_metric)

        # --- Stress ---
        N = membrane_stress(material, E)

        # Quadrature weight
        w = getweight(scv, q) * geom.detJ

        # --- Loop over nodes ---
        nn = getnnodes(scv)

        for I in 1:nn

            dN_dξ1 = shape_gradient(scv.cellvalues, I, q)[1]
            dN_dξ2 = shape_gradient(scv.cellvalues, I, q)[2]

            # contraction:
            # ∂α N_I * N_{αβ} * a_β

            # First build vector:
            v =
                dN_dξ1 * (N[1,1]*a1 + N[1,2]*a2) +
                dN_dξ2 * (N[1,2]*a1 + N[2,2]*a2)

            # assemble into residual
            re[3I-2:3I] .+= v * w
        end
    end
end

function element_membrane_tangent!(Ke, scv, u_e, material)

    fill!(Ke, 0.0)

    nqp = getnquadpoints(scv)
    nn  = getnnodes(scv)

    for q in 1:nqp

        geom = update!(scv, q, u_e)

        a1 = geom.a1
        a2 = geom.a2

        A_metric = geom.A_metric
        a_metric = geom.a_metric

        E = 0.5 * (a_metric - A_metric)

        N, C = membrane_stress_and_tangent(material, E)

        w = getweight(scv, q) * geom.detJ

        for I in 1:nn

            ∂NI1 = shape_gradient(scv.cellvalues, I, q)[1]
            ∂NI2 = shape_gradient(scv.cellvalues, I, q)[2]

            for J in 1:nn

                ∂NJ1 = shape_gradient(scv.cellvalues, J, q)[1]
                ∂NJ2 = shape_gradient(scv.cellvalues, J, q)[2]

                # -----------------
                # Geometric term
                # -----------------

                geo_scalar =
                    ∂NI1*(N[1,1]*∂NJ1 + N[1,2]*∂NJ2) +
                    ∂NI2*(N[1,2]*∂NJ1 + N[2,2]*∂NJ2)

                Kgeo = geo_scalar * I₃

                # -----------------
                # Material term
                # -----------------

                # Build δE components for J-direction
                # γδ indices = (1,1), (1,2), (2,2)

                # compute symmetric strain variation directions

                B11 = ∂NJ1 * a1
                B22 = ∂NJ2 * a2
                B12 = 0.5*(∂NJ1*a2 + ∂NJ2*a1)

                # contract with constitutive tensor C
                # (write in Voigt for simplicity)

                # resulting vector in ℝ³:
                Kmat = compute_material_block(
                    ∂NI1, ∂NI2,
                    B11, B12, B22,
                    a1, a2,
                    C
                )

                # assemble
                Ke[3I-2:3I, 3J-2:3J] .+= (Kgeo + Kmat) * w
            end
        end
    end
end