
#include <iostream>
#include <stdlib.h>

#include "csr_matrix.h"
#include "solve_cusparse.h"
#include "solve_cusp.h"
#include "solve_wsmp.h"

using namespace std;

int main(int argc, char **argv) {
    if (argc < 3) {
        cout << "usage: csr_matrix_file matrix_market_format" << endl;
        exit(-1);
    }

    csr_matrix matrix(argv[1]);
#ifdef CUDA_FOUND
    solve_cusp(matrix, argv[2]);
    solve_cusparse(matrix, argv[2]);
#endif
    solve_wsmp(matrix, argv[2]);


    return 0;
}
