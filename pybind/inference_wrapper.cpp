//File name: functions · wrapper.cpp
#include <pybind11/pybind11.h>  
#include "inference.h"  
  
namespace py = pybind11;  
  
PYBIND11_MODULE(inference, m){  
    m.def("predict",&predict, "predict");
    m.def("affine_batched",&affine_batched, "affine_batched");
    m.def("relu",&relu, "relu");
}
