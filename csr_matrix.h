#ifndef CSR_MATRIX_H
#define CSR_MATRIX_H

#include <assert.h>
#include <stdio.h>

class csr_matrix {
public:
    csr_matrix(const char *fname) {
        load(fname);
    }

    ~csr_matrix() {
        delete[] x;
        delete[] b;
        delete[] values;
        delete[] columns;
        delete[] row_index;
    }

    int nnz, n;
    int *row_index, *columns;
    double *values;
    double *b, *x;

    void to_c_indices() {
        for (int i = 0; i < nnz; i++) {
            columns[i]--;
            if (i < n+1) {
                row_index[i]--;
            }
        }
    }

private:
    void load(const char *fname) {
        FILE *dump_file = fopen(fname, "ro");
        fscanf(dump_file, "%d %d\n", &n, &nnz);
        assert(n > 0);
        assert(nnz > 0);
        assert(nnz >= n);
        row_index = new int[n+1];
        columns = new int[nnz];
        values = new double[nnz];
        b = new double[n];
        x = new double[n];

        for (int i = 0; i < n+1; i++) {
            fscanf(dump_file, "%d\n", &row_index[i]);
        }
        fscanf(dump_file, "\n");
        for (int i = 0; i < nnz; i++) {
            fscanf(dump_file, "%d %lf\n", &columns[i], &values[i]);
        }
        fscanf(dump_file, "\n");
        for (int i = 0; i < n; i++) {
            fscanf(dump_file, "%lf %lf\n", &b[i], &x[i]);
        }
    }
};

#endif // CSR_MATRIX_H
