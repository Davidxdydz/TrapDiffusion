def test_concentration_constant():
    import numpy as np
    from models.analytical import SingleOccupationSingleIsotope

    for _ in range(100):
        analytical_model = SingleOccupationSingleIsotope(2, 1)
        c0 = analytical_model.initial_values()
        ts, cs = analytical_model.solve(c0, 50)
        cs *= analytical_model.c_S_T[:, None]
        total = np.sum(cs, axis=0)
        assert np.allclose(total - 1, 0)
