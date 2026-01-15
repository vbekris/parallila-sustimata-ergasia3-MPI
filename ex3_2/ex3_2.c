#include <stdio.h>
#include <stdlib.h>
#include <string.h> // Για memcpy
#include <mpi.h>
#include "timer.h" 

/* --- Δομή CSR --- */
typedef struct {
    double *values;
    int *col_ind;
    int *row_ptr;
    int n; 
    int nnz;
} csr_t;

/* --- Μετατροπή Dense -> CSR --- */
csr_t dense2csr(double *A_dense, int n) {
    csr_t mat;
    mat.n = n;
    int count = 0;
    for (int i = 0; i < n * n; i++) if (A_dense[i] != 0.0) count++;
    mat.nnz = count;

    mat.values = (double*) malloc(count * sizeof(double));
    mat.col_ind = (int*) malloc(count * sizeof(int));
    mat.row_ptr = (int*) malloc((n + 1) * sizeof(int));

    int current_nnz = 0;
    mat.row_ptr[0] = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double val = A_dense[i*n + j];
            if (val != 0.0) {
                mat.values[current_nnz] = val;
                mat.col_ind[current_nnz] = j;
                current_nnz++;
            }
        }
        mat.row_ptr[i+1] = current_nnz;
    }
    return mat;
}

void free_csr(csr_t *mat) {
    if (mat->values) free(mat->values);
    if (mat->col_ind) free(mat->col_ind);
    if (mat->row_ptr) free(mat->row_ptr);
}



int main(int argc, char* argv[]) {
    int my_rank, comm_sz;
    int n, iters;
    double sparsity;

// --- ΔΙΟΡΘΩΣΗ ΕΔΩ: Αρχικοποίηση όλων στο 0.0 ---
    double t_csr_create_start = 0.0, t_csr_create_end = 0.0;
    double t_csr_comm_start = 0.0, t_csr_comm_end = 0.0;
    double t_csr_calc_start = 0.0, t_csr_calc_end = 0.0;
    double t_dense_comm_start = 0.0, t_dense_comm_end = 0.0; 
    double t_dense_calc_start = 0.0, t_dense_calc_end = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (argc != 4) {
        if (my_rank == 0) printf("Usage: %s <n> <sparsity> <iters>\n", argv[0]);
        MPI_Finalize(); return 0;
    }

    n = atoi(argv[1]);
    sparsity = atof(argv[2]);
    iters = atoi(argv[3]);

    if (n % comm_sz != 0) {
        if (my_rank == 0) printf("Error: n must be divisible by P\n");
        MPI_Finalize(); return 0;
    }

    /* =========================================
       PHASE 1: GENERATION (Rank 0)
       ========================================= */
    double *A_dense_global = NULL;
    double *x_global = NULL;   // Το αρχικό x
    double *x = NULL;          // Το τοπικό x (που θα αλλάζει)
    double *x_copy = NULL;     // Αντίγραφο για reset πριν το Dense
    csr_t global_csr;

    // Όλοι θέλουν χώρο για το x
    x = (double*) malloc(n * sizeof(double));
    x_copy = (double*) malloc(n * sizeof(double));

    if (my_rank == 0) {
        printf("Master: Generating N=%d, Sparsity=%.2f...\n", n, sparsity);
        A_dense_global = (double*) malloc(n * n * sizeof(double));
        x_global = (double*) malloc(n * sizeof(double));

        srand(42);
        for(int i=0; i<n*n; i++) {
            double r = (double)rand() / RAND_MAX;
            A_dense_global[i] = (r > sparsity) ? ((rand()%10)+1) : 0.0;
        }
        for(int i=0; i<n; i++) x_global[i] = 1.0;

        // Αντιγραφή στο local x του Master
        memcpy(x, x_global, n * sizeof(double));

        // (i) Μέτρηση Χρόνου Κατασκευής CSR
        GET_TIME(t_csr_create_start);
        global_csr = dense2csr(A_dense_global, n);
        GET_TIME(t_csr_create_end);
    }

    // Διανομή του αρχικού x σε όλους (για να ξεκινήσουν ίδια)
    // Αν ο Master το έχει στο 'x', το Bcast το στέλνει στους άλλους στο 'x'
    MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // Κρατάμε backup για το Dense part
    memcpy(x_copy, x, n * sizeof(double));


    /* =========================================
       PHASE 2: CSR DISTRIBUTION & CALC
       ========================================= */
    int local_n = n / comm_sz;
    int local_nnz;
    int *scounts = NULL, *displs = NULL;

    if (my_rank == 0) {
        scounts = malloc(comm_sz * sizeof(int));
        displs = malloc(comm_sz * sizeof(int));
        for (int i=0; i<comm_sz; i++) {
            int start = i * local_n;
            int end = (i+1) * local_n;
            scounts[i] = global_csr.row_ptr[end] - global_csr.row_ptr[start];
            displs[i] = global_csr.row_ptr[start];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    GET_TIME(t_csr_comm_start);

    // 1. Send counts
    MPI_Scatter(scounts, 1, MPI_INT, &local_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 2. Allocate Local CSR
    csr_t local_csr;
    local_csr.n = local_n;
    local_csr.nnz = local_nnz;
    local_csr.values = malloc(local_nnz * sizeof(double));
    local_csr.col_ind = malloc(local_nnz * sizeof(int));
    local_csr.row_ptr = malloc((local_n + 1) * sizeof(int));

    // 3. Send Data (Scatterv)
    MPI_Scatterv(my_rank==0 ? global_csr.values : NULL, scounts, displs, MPI_DOUBLE, 
                 local_csr.values, local_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(my_rank==0 ? global_csr.col_ind : NULL, scounts, displs, MPI_INT, 
                 local_csr.col_ind, local_nnz, MPI_INT, 0, MPI_COMM_WORLD);
    
    // 4. Send Row Ptrs
    MPI_Scatter(my_rank==0 ? global_csr.row_ptr : NULL, local_n, MPI_INT,
                local_csr.row_ptr, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    // 5. Normalize Row Ptrs
    int start_idx = local_csr.row_ptr[0];
    for(int i=0; i<local_n; i++) local_csr.row_ptr[i] -= start_idx;
    local_csr.row_ptr[local_n] = local_nnz;

    MPI_Barrier(MPI_COMM_WORLD);
    GET_TIME(t_csr_comm_end);

    // 6. CSR Loop
    double *local_y = calloc(local_n, sizeof(double)); // Result buffer

    MPI_Barrier(MPI_COMM_WORLD);
    GET_TIME(t_csr_calc_start);

    for (int iter = 0; iter < iters; iter++) {
        // Kernel
        for (int i = 0; i < local_n; i++) {
            double sum = 0.0;
            for (int j = local_csr.row_ptr[i]; j < local_csr.row_ptr[i+1]; j++) {
                sum += local_csr.values[j] * x[local_csr.col_ind[j]];
            }
            local_y[i] = sum;
        }
        // Update x for next iter
        MPI_Allgather(local_y, local_n, MPI_DOUBLE, x, local_n, MPI_DOUBLE, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    GET_TIME(t_csr_calc_end);

    // *Cleanup CSR Local* (Για να μην πιάσουμε μνήμη στο Dense)
    free(local_csr.values); free(local_csr.col_ind); free(local_csr.row_ptr);

    /* =========================================
       PHASE 3: DENSE PARALLEL DISTRIBUTION & CALC
       ========================================= */
    
    // 1. Reset x (Για δίκαιη σύγκριση)
    memcpy(x, x_copy, n * sizeof(double));

    // 2. Allocate Local Dense Matrix
    // Κάθε διεργασία παίρνει local_n γραμμές, άρα local_n * N στοιχεία
    double *local_A_dense = malloc(local_n * n * sizeof(double));

    MPI_Barrier(MPI_COMM_WORLD);
    GET_TIME(t_dense_comm_start);

    // 3. Scatter Dense Matrix
    // Εδώ χρησιμοποιούμε απλό MPI_Scatter γιατί τα κομμάτια είναι ίσα (local_n * n)
    MPI_Scatter(A_dense_global, local_n * n, MPI_DOUBLE,
                local_A_dense, local_n * n, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    GET_TIME(t_dense_comm_end);

    // 4. Dense Loop
    MPI_Barrier(MPI_COMM_WORLD);
    GET_TIME(t_dense_calc_start);

    for (int iter = 0; iter < iters; iter++) {
        // Dense Kernel
        for (int i = 0; i < local_n; i++) {
            double sum = 0.0;
            // Διασχίζουμε ΟΛΗ τη γραμμή (N στοιχεία)
            for (int j = 0; j < n; j++) {
                sum += local_A_dense[i*n + j] * x[j];
            }
            local_y[i] = sum;
        }
        // Update x for next iter
        MPI_Allgather(local_y, local_n, MPI_DOUBLE, x, local_n, MPI_DOUBLE, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    GET_TIME(t_dense_calc_end);

    /* =========================================
       PHASE 4: RESULTS & CLEANUP
       ========================================= */
    
    // Συλλογή τελικού αποτελέσματος (προαιρετικά, για την πληρότητα της εκφώνησης)
    // "Η διεργασία 0 συγκεντρώνει το τελικό αποτέλεσμα και το τυπώνει"
    double *final_result = NULL;
    if (my_rank == 0) final_result = malloc(n * sizeof(double));
    
    // Μαζεύουμε το τελικό y (που είναι ουσιαστικά το x της τελευταίας επανάληψης)
    // Μπορούμε απλά να πάρουμε το x, αφού το Allgather μόλις έγινε.
    if (my_rank == 0) memcpy(final_result, x, n * sizeof(double));

    if (my_rank == 0) {
        double csr_total = (t_csr_create_end - t_csr_create_start) + // (i)
                           (t_csr_comm_end - t_csr_comm_start) +     // (ii)
                           (t_csr_calc_end - t_csr_calc_start);      // (iii)
        
        // Ο Συνολικός χρόνος Dense περιλαμβάνει την αποστολή (comm) και τον υπολογισμό (calc)
        double dense_total = (t_dense_comm_end - t_dense_comm_start) + 
                             (t_dense_calc_end - t_dense_calc_start);

        printf("\n=== RESULTS (N=%d, Sparsity=%.2f, P=%d, Iters=%d) ===\n", n, sparsity, comm_sz, iters);
        printf("(i)   CSR Creation Time:      %e sec\n", t_csr_create_end - t_csr_create_start);
        printf("(ii)  CSR Comm Time (Distr):  %e sec\n", t_csr_comm_end - t_csr_comm_start);
        printf("(iii) CSR Calc Time:          %e sec\n", t_csr_calc_end - t_csr_calc_start);
        printf("(iv)  Total CSR Time:         %e sec\n", csr_total);
        printf("(v)   Total Dense Time (MPI): %e sec\n", dense_total);
        printf("----------------------------------------------------\n");
        printf("Dense Comm Time:              %e sec\n", t_dense_comm_end - t_dense_comm_start);
        printf("Dense Calc Time:              %e sec\n", t_dense_calc_end - t_dense_calc_start);

        // Προαιρετική εκτύπωση αποτελέσματος για μικρά N
        if (n <= 10) {
            printf("Final Result Vector: ");
            for(int k=0; k<n; k++) printf("%.1f ", final_result[k]);
            printf("\n");
        }
        
        free(A_dense_global); free(x_global); free(final_result); free_csr(&global_csr);
        free(scounts); free(displs);
    }

    free(x); free(x_copy); free(local_y); free(local_A_dense);
    MPI_Finalize();
    return 0;
}