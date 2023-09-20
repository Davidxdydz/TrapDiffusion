def test_concentration_constant():
    import numpy as np
    from models.analytical.trapdiffusion import SingleOccupationSingleIsotope
    for _ in range(100):
        analytical_model = SingleOccupationSingleIsotope(2,1)
        analytical_model.solve(analytical_model.c)
        corrected = analytical_model.sol.y
        corrected *= analytical_model.c_S_T[:,None]
        total = np.sum(corrected, axis=0)
        assert np.allclose(total - 1,0)
