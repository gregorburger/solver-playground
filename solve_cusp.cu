#include <iostream>
using namespace std;
#include <assert.h>

#include "csr_matrix.h"
#include "cusp/csr_matrix.h"
#include "cusp/verify.h"
#include "cusp/krylov/cg.h"
#include "cusp/krylov/bicgstab.h"
#include "cusp/krylov/gmres.h"
#include "cusp/io/matrix_market.h"
#include "cusp/print.h"
#include "cusp/precond/diagonal.h"

void solve_cusp(csr_matrix &matrix, const char *fname) {
    cusp::csr_matrix<int, float, cusp::host_memory> A;
    cusp::io::read_matrix_market_file(A, fname);
    //cusp::print(dev_A);

    cusp::csr_matrix<int, float, cusp::device_memory> dev_A(A);
    assert(cusp::is_valid_matrix(dev_A, std::cout));

    cusp::array1d<float, cusp::host_memory> b(matrix.n);
    for (int i = 0; i < matrix.n; i++) {
        b[i] = matrix.b[i];
    }

    cusp::array1d<float, cusp::device_memory> dev_b(b);
    cusp::array1d<float, cusp::device_memory> dev_x(b.size(), 0);

    cusp::verbose_monitor<float> monitor(dev_b, 200000, 0);
    //cusp::precond::diagonal<float, cusp::device_memory> precond(dev_A);
    cusp::krylov::bicgstab(dev_A, dev_x, dev_b, monitor);
    cout << monitor.residual_norm() << endl;
    //assert(monitor.converged());
    cusp::array1d<float, cusp::host_memory> x(dev_x);

    for (int i = 0; i < matrix.n; i++) {
        printf("%30.7f %30.7f\n", x[i], matrix.x[i]);
    }
}
