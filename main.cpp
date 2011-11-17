
#include <iostream>
#include <stdlib.h>

#include "csr_matrix.h"
#include "solve_cusparse.h"
#include "solve_cusp.h"

using namespace std;

int main(int argc, char **argv) {
    if (argc < 3) {
        cout << "usage: matrix_market_format csr_matrix_file" << endl;
        exit(-1);
    }

    csr_matrix matrix(argv[1]);
    //cusp_solve(matrix, argv[2]);
    solve_cusparse(matrix, argv[2]);

    return 0;
}
