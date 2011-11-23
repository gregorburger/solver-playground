#include "solve_wsmp.h"
#include "unistd.h"

#include "csr_matrix.h"

extern "C" {
void wsmp_initialize_();
double wsmprtc_();
void wsetmaxthrds_(int *);
void wscalz_(int *n, int *ia, int *ja,
            int *options, int *perm, int *invp,
            int *nnzl, int *wspace, int *aux,
            int *naux, int *info);

void  wscchf_(int *n, int *ia, int *ja,
              double *avals, int *perm, int *invp,
              int *aux, int *naux, int *info);

void wsslv_(int *n, int *perm, int *invp,
            double *b, int *ldb, int *nrhs,
            int *niter, int *aux, int *naux);
}


void solve_wsmp(csr_matrix &m, const char *) {
    wsmp_initialize_();
    int options[5] = {0, 0, 0, 0, 0};
    int *perm = new int[m.n];
    int *invp = new int[m.n];
    int nnzl, wspace, aux, naux, info;

    naux = 0;
    aux = 0;

    wscalz_(&m.n, m.row_index, m.columns,
           options, perm, invp,
           &nnzl, &wspace, &aux, &naux, &info);

    assert(info == 0);

    double before = wsmprtc_();

    wscchf_(&m.n, m.row_index, m.columns, m.values,
            perm, invp, &aux, &naux, &info);

    assert(info == 0);

    int ldb = m.n;
    int nrhs = 1;
    int niter = 0;

    wsslv_(&m.n, perm, invp, m.b, &ldb, &nrhs, &niter, &aux, &naux);

    double after = wsmprtc_();

    printf("solving took %f seconds\n", after - before);
}
