#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "xtensor/xmath.hpp"              // xtensor import for the C++ universal functions
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"     // Numpy bindings

#include "biot_savart.h"

#include "linking_number.hh"
typedef LK::LinkingNumber<double> LK_class;

int add(int i, int j) {
    return i + j;
}

namespace py = pybind11;

PYBIND11_MODULE(cppplasmaopt, m) {
    xt::import_numpy();

    m.def("add", &add, R"pbdoc(
        Add two numbers
        Some other explanation about the add function.
    )pbdoc");

    m.def("biot_savart_all",               & biot_savart_all);
    m.def("biot_savart_B_only",            & biot_savart_B_only);
    m.def("biot_savart_by_dcoilcoeff_all", & biot_savart_by_dcoilcoeff_all);
    
    m.def("biot_savart_B",           &biot_savart_B);
    m.def("biot_savart_dB_by_dX",    &biot_savart_dB_by_dX);
    m.def("biot_savart_d2B_by_dXdX", &biot_savart_d2B_by_dXdX);

    m.def("biot_savart_dB_by_dcoilcoeff",    & biot_savart_dB_by_dcoilcoeff);
    m.def("biot_savart_d2B_by_dXdcoilcoeff", & biot_savart_d2B_by_dXdcoilcoeff);

    m.def("ln", [](Array& A, Array& B) {
            LK_class lk(2);
            int nseg1 = A.shape(0);
            int nseg2 = B.shape(0);
            
            double c1[10000][3];
            double c2[10000][3];
            for(int i; i < nseg1; i++){
                c1[i][0] = A(i, 0);
                c1[i][1] = A(i, 1);
                c1[i][2] = A(i, 2);
            }
    
            for(int i; i < nseg2; i++){
                c2[i][0] = B(i, 0);
                c2[i][1] = B(i, 1);
                c2[i][2] = B(i, 2);
            }
    
            return lk.eval(c1, nseg1, c2, nseg2) ;
        });

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
