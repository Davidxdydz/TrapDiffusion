#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>

using namespace std;
using namespace pybind11::literals;
namespace py = pybind11;

void predict_cpp(vector<float *> weights, vector<float *> biases, float *input, float *output, vector<size_t> rows, vector<size_t> columns, size_t batch_size);

py::array_t<float> predict(vector<py::array_t<float>> layer_weights, vector<py::array_t<float>> layer_bias, py::array_t<float> input);

void relu_cpp(float *x, size_t count);

py::array_t<float> relu(py::array_t<float> x);

void affine_batched_cpp(float *A, float *b, float *x, float *y, size_t rows, size_t columns, size_t batch_size);

py::array_t<float> affine_batched(py::array_t<float> A, py::array_t<float> b, py::array_t<float> x);
