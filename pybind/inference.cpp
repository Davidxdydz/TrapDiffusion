#include "inference.h"

void predict_cpp(vector<float *> weights, vector<float *> biases, float *input, float *output, vector<size_t> rows, vector<size_t> columns, size_t batch_size)
{
    size_t depth = weights.size();
    size_t largest_tmp = 0;
    for (size_t row : rows)
    {
        if (row > largest_tmp)
            largest_tmp = row;
    }
    largest_tmp *= batch_size;
    float *buffers[] = {new float[largest_tmp], new float[largest_tmp]};
    for (int d = 0; d < depth; d++)
    {
        float *input_buff = buffers[d % 2];
        float *output_buff = buffers[(d + 1) % 2];
        if (d == 0)
            input_buff = input;
        if (d == depth - 1)
            output_buff = output;
        affine_batched_cpp(weights[d], biases[d], input_buff, output_buff, rows[d], columns[d], batch_size);
        relu_cpp(output_buff, rows[d] * batch_size);
    }

    delete[] buffers[0];
    delete[] buffers[1];
}

py::array_t<float> predict(vector<py::array_t<float>> layer_weights, vector<py::array_t<float>> layer_bias, py::array_t<float> input)
{
    vector<float *> weights;
    vector<float *> biases;
    vector<size_t> rows;
    vector<size_t> columns;

    py::buffer_info input_buf = input.request();
    auto shape = input_buf.shape;
    size_t batch_size = shape[0];
    size_t depth = layer_weights.size();
    for (int d = 0; d < depth; d++)
    {
        py::buffer_info weights_buf = layer_weights[d].request();
        py::buffer_info bias_buf = layer_bias[d].request();
        rows.push_back(weights_buf.shape[0]);
        columns.push_back(weights_buf.shape[1]);
        weights.push_back((float *)weights_buf.ptr);
        biases.push_back((float *)bias_buf.ptr);
    }

    py::array_t<float> result = py::array_t<float>({batch_size, rows[depth - 1]});
    predict_cpp(weights, biases, (float *)input_buf.ptr, (float *)result.request().ptr, rows, columns, batch_size);
    return result;
}

void relu_cpp(float *x, size_t count)
{
    // in place
    for (int i = 0; i < count; i++)
    {
        if (x[i] < 0)
            x[i] = 0;
    }
}
void affine_batched_cpp(float *A, float *b, float *x, float *y, size_t rows, size_t columns, size_t batch_size)
{
    // first dimension in x is batch
    for (int batch = 0; batch < batch_size; batch++)
    {
        for (int i = 0; i < rows; i++)
        {
            auto current = rows * batch + i;
            y[current] = 0;
            for (int j = 0; j < columns; j++)
            {
                y[current] += A[i * columns + j] * x[columns * batch + j];
            }
            y[current] += b[i];
        }
    }
}

py::array_t<float> relu(py::array_t<float> x)
{
    py::buffer_info x_buf = x.request();
    py::array_t<float> result = py::array_t<float>(x_buf.shape);
    memcpy(result.request().ptr, x_buf.ptr, x_buf.size * sizeof(float));
    relu_cpp((float *)result.request().ptr, x_buf.size);
    return result;
}

py::array_t<float> affine_batched(py::array_t<float> A, py::array_t<float> b, py::array_t<float> x)
{
    py::buffer_info A_buf = A.request();
    py::buffer_info x_buf = x.request();
    py::buffer_info b_buf = b.request();
    if (A_buf.ndim != 2 || x_buf.ndim != 2 || b_buf.ndim != 1)
        throw std::runtime_error("Number of dimensions must be two for A and x and one for b.");
    if (A_buf.shape[1] != x_buf.shape[1])
        throw std::runtime_error("Second dimension of A must match first dimension of x.");
    if (A_buf.shape[0] != b_buf.shape[0])
        throw std::runtime_error("First dimension of A must match dimension of b.");
    size_t batch_size = x_buf.shape[0];
    size_t rows = A_buf.shape[0];
    size_t columns = A_buf.shape[1];
    py::array_t<float> result = py::array_t<float>({batch_size, rows});
    affine_batched_cpp((float *)A_buf.ptr, (float *)b_buf.ptr, (float *)x_buf.ptr, (float *)result.request().ptr, rows, columns, batch_size);
    return result;
}