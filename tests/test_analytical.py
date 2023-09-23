def test_concentration_constant():
    import numpy as np
    from models.analytical import (
        SingleOccupationSingleIsotope,
        MultiOccupationMultiIsotope,
    )

    for model in [SingleOccupationSingleIsotope, MultiOccupationMultiIsotope]:
        for _ in range(100):
            analytical_model = model()
            c0 = analytical_model.initial_values()
            ts, cs = analytical_model.solve(c0)
            cs *= analytical_model.correction_factors()[:, None]
            total = np.sum(cs, axis=0)
            assert np.allclose(
                total, 1
            ), f"{analytical_model.name} does not conserve mass"
