def test_affine():
    import numpy as np
    import inference
    n = 50
    batch = 50
    A = np.random.random((n,2*n))
    b = np.random.random((n,))
    x = np.random.random((batch,2*n))
    cpp = inference.affine_batched(A,b,x)
    for i in range(batch):
        py = A @ x[i] + b
        assert np.allclose(cpp[i],py)

def test_relu():
    import numpy as np
    import inference
    a = np.random.uniform(-1,1,(50,50))
    cpp = inference.relu(a)
    py = np.maximum(a,0)
    assert np.allclose(cpp,py)
    a = np.random.uniform(-1,1,(50,))
    cpp = inference.relu(a)
    py = np.maximum(a,0)
    assert np.allclose(cpp,py)

def test_prediction():
    import numpy as np
    from models.analytical.trapdiffusion import SingleOccupationSingleIsotope
    import inference
    import keras_core as keras
    basic_model = keras.models.load_model('trained_models/basic.keras')
    weightsT = []
    biases = []
    for layer in basic_model.layers:
        weightsT.append(np.ascontiguousarray(layer.get_weights()[0].T))
        biases.append(layer.get_weights()[1])
    analytical_model = SingleOccupationSingleIsotope(2,1)
    c_init = analytical_model.initial_values()
    ts, cs = analytical_model.solve(c_init,50)
    inputs = np.repeat([c_init],len(ts),axis=0)
    inputs = np.append(ts.reshape(-1,1),inputs,axis=1)
    predictions1 = inference.predict(weightsT, biases, inputs)
    predictions2 = basic_model.predict(inputs)
    assert np.allclose(predictions1,predictions2)