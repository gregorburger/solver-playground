#include "solve_cusparse.h"
#include <cusparse_v2.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include <iostream>
using namespace std;

//CudasparseSafeCall
#define CSC(func) { \
    cusparseStatus_t status = func; \
    if (status != CUSPARSE_STATUS_SUCCESS) { \
        cout << "error calling: " << #func << endl; \
    } \
}

#define CUDA_SC(func) { \
    cudaError_t error; \
    error = func; \
    if (error != cudaSuccess) {\
        cout << "error calling: " << #func << endl; \
    }\
}

void solve_cusparse(csr_matrix &matrix, const char *finput) {
    cusparseHandle_t handle;
    CSC(cusparseCreate(&handle));

    cusparseMatDescr_t desc;
    CSC(cusparseCreateMatDescr(&desc));

    CSC(cusparseSetMatDiagType(desc, CUSPARSE_DIAG_TYPE_NON_UNIT));
    CSC(cusparseSetMatFillMode(desc, CUSPARSE_FILL_MODE_LOWER))
    CSC(cusparseSetMatIndexBase(desc, CUSPARSE_INDEX_BASE_ONE))
    CSC(cusparseSetMatType(desc, CUSPARSE_MATRIX_TYPE_TRIANGULAR));

    double *values, *x, *y;
    int *row_index, *columns;

    CUDA_SC(cudaMalloc(&values,     sizeof(double) *    matrix.nnz));
    CUDA_SC(cudaMalloc(&x,          sizeof(double) *    matrix.n));
    CUDA_SC(cudaMalloc(&y,          sizeof(double) *    matrix.n));
    CUDA_SC(cudaMalloc(&row_index,  sizeof(int) *       (matrix.n+1)));
    CUDA_SC(cudaMalloc(&columns,    sizeof(int) *       matrix.nnz));


    CUDA_SC(cudaMemcpy(values, matrix.values,       matrix.nnz * sizeof(double),    cudaMemcpyHostToDevice));
    CUDA_SC(cudaMemcpy(x, matrix.b,                 matrix.n * sizeof(double),    cudaMemcpyHostToDevice));
    CUDA_SC(cudaMemcpy(columns, matrix.columns,     matrix.nnz * sizeof(int),       cudaMemcpyHostToDevice));
    CUDA_SC(cudaMemcpy(row_index, matrix.row_index, (matrix.n+1) * sizeof(int),     cudaMemcpyHostToDevice));


    cusparseSolveAnalysisInfo_t info;
    cusparseCreateSolveAnalysisInfo(&info);

    cusparseStatus_t status;

    status = cusparseDcsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrix.n, matrix.nnz, desc, values, row_index, columns, info);
    assert(status == CUSPARSE_STATUS_SUCCESS);

    double alpha = 1.0;

    status = cusparseDcsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrix.n, &alpha, desc, values, row_index, columns, info, x, y);
    assert(status == CUSPARSE_STATUS_SUCCESS);

    status = cusparseDcsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrix.n, &alpha, desc, values, row_index, columns, info, x, y);
    assert(status == CUSPARSE_STATUS_SUCCESS);

    status = cusparseDcsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrix.n, &alpha, desc, values, row_index, columns, info, x, y);
    assert(status == CUSPARSE_STATUS_SUCCESS);

    status = cusparseDcsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrix.n, &alpha, desc, values, row_index, columns, info, x, y);
    assert(status == CUSPARSE_STATUS_SUCCESS);

    double *host_y = new double[matrix.n];

    CUDA_SC(cudaMemcpy(host_y, y, sizeof(double) * matrix.n, cudaMemcpyDeviceToHost));

    for (int i = 0; i < matrix.n; i++) {
        printf("%f %f\n", host_y[i], matrix.x[i]);
    }

    //clean up
    cusparseDestroySolveAnalysisInfo(info);

    CUDA_SC(cudaFree(columns));
    CUDA_SC(cudaFree(row_index));
    CUDA_SC(cudaFree(y));
    CUDA_SC(cudaFree(x));
    CUDA_SC(cudaFree(values));

    CSC(cusparseDestroyMatDescr(desc));

    CSC(cusparseDestroy(handle));
}
