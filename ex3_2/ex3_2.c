#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "timer.h" 

/* --- Δομή CSR (Compressed Sparse Row) --- */
typedef struct {
    double *values;   // Μη-μηδενικές τιμές
    int *col_ind;     // Δείκτες στήλης για κάθε τιμή
    int *row_ptr;     // Δείκτες αρχής κάθε γραμμής
    int n;            // Αριθμός γραμμών
    int nnz;          // Πλήθος μη-μηδενικών στοιχείων (Number of Non-Zeros)
} csr_t;

/* --- Μετατροπή Dense -> CSR --- */
// Υλοποιείται σε 2 περάσματα: 
// 1. Καταμέτρηση NNZ για δέσμευση μνήμης.
// 2. Γέμισμα των πινάκων values, col_ind, row_ptr.
csr_t dense2csr(double *A_dense, int n) {
    csr_t mat;
    mat.n = n;
    int count = 0;
    
    // Pass 1: Μέτρηση μη-μηδενικών
    for (int i = 0; i < n * n; i++) if (A_dense[i] != 0.0) count++;
    mat.nnz = count;

    // Δέσμευση μνήμης για CSR
    mat.values = (double*) malloc(count * sizeof(double));
    mat.col_ind = (int*) malloc(count * sizeof(int));
    mat.row_ptr = (int*) malloc((n + 1) * sizeof(int));

    // Pass 2: Γέμισμα δομής
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
        mat.row_ptr[i+1] = current_nnz; // Τέλος τρέχουσας γραμμής / Αρχή επόμενης
    }
    return mat;
}

// Συνάρτηση αποδέσμευσης CSR
void free_csr(csr_t *mat) {
    if (mat->values) free(mat->values);
    if (mat->col_ind) free(mat->col_ind);
    if (mat->row_ptr) free(mat->row_ptr);
}

int main(int argc, char* argv[]) {
    int my_rank, comm_sz;
    int n, iters;
    double sparsity;

    // Μεταβλητές χρονομέτρησης
    double t_csr_create_start = 0.0, t_csr_create_end = 0.0;
    double t_csr_comm_start = 0.0, t_csr_comm_end = 0.0;
    double t_csr_calc_start = 0.0, t_csr_calc_end = 0.0;
    double t_dense_comm_start = 0.0, t_dense_comm_end = 0.0; 
    double t_dense_calc_start = 0.0, t_dense_calc_end = 0.0;

    // MPI Initialization
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    // Έλεγχος Ορισμάτων
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

    /* ======================================================
       PHASE 1: GENERATION (Rank 0 Only)
       ====================================================== */
    double *A_dense_global = NULL;
    double *x_global = NULL;   
    double *x = NULL;          // Τοπικό διάνυσμα x (ανανεώνεται σε κάθε iter)
    double *x_copy = NULL;     // Backup για χρήση στο Dense μέρος
    csr_t global_csr;

    // Δέσμευση χώρου για το διάνυσμα x σε όλες τις διεργασίες
    x = (double*) malloc(n * sizeof(double));
    x_copy = (double*) malloc(n * sizeof(double));

    if (my_rank == 0) {
        printf("Master: Generating N=%d, Sparsity=%.2f...\n", n, sparsity);
        A_dense_global = (double*) malloc(n * n * sizeof(double));
        x_global = (double*) malloc(n * sizeof(double));

        // Τυχαία αρχικοποίηση με βάση το sparsity
        srand(42);
        for(int i=0; i<n*n; i++) {
            double r = (double)rand() / RAND_MAX;
            A_dense_global[i] = (r > sparsity) ? ((rand()%10)+1) : 0.0;
        }
        for(int i=0; i<n; i++) x_global[i] = 1.0; // Αρχικοποίηση x με 1

        memcpy(x, x_global, n * sizeof(double));

        // (i) Κατασκευή CSR και Χρονομέτρηση
        GET_TIME(t_csr_create_start);
        global_csr = dense2csr(A_dense_global, n);
        GET_TIME(t_csr_create_end);
    }

    // Διανομή του αρχικού διανύσματος x σε όλους
    MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    memcpy(x_copy, x, n * sizeof(double)); // Backup για το Dense πείραμα


    /* ======================================================
       PHASE 2: CSR DISTRIBUTION & CALCULATION
       ====================================================== */
    int local_n = n / comm_sz; // Γραμμές ανά διεργασία
    int local_nnz;             // Μη-μηδενικά ανά διεργασία (διαφέρει στον καθένα)
    
    // Πίνακες για το Scatterv (διαχειρίζονται τα άνισα μεγέθη δεδομένων)
    int *scounts = NULL, *displs = NULL;

    if (my_rank == 0) {
        scounts = malloc(comm_sz * sizeof(int));
        displs = malloc(comm_sz * sizeof(int));
        
        // Υπολογισμός κατανομής φορτίου (nnz) ανά διεργασία
        for (int i=0; i<comm_sz; i++) {
            int start = i * local_n;
            int end = (i+1) * local_n;
            scounts[i] = global_csr.row_ptr[end] - global_csr.row_ptr[start];
            displs[i] = global_csr.row_ptr[start];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    GET_TIME(t_csr_comm_start);

    // 1. Ενημέρωση διεργασιών για το μέγεθος δεδομένων που θα λάβουν (local_nnz)
    MPI_Scatter(scounts, 1, MPI_INT, &local_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 2. Δέσμευση Τοπικής Μνήμης CSR
    csr_t local_csr;
    local_csr.n = local_n;
    local_csr.nnz = local_nnz;
    local_csr.values = malloc(local_nnz * sizeof(double));
    local_csr.col_ind = malloc(local_nnz * sizeof(int));
    local_csr.row_ptr = malloc((local_n + 1) * sizeof(int));

    // 3. Διανομή δεδομένων (Χρήση Scatterv λόγω ανισοκατανομής των μηδενικών)
    MPI_Scatterv(my_rank==0 ? global_csr.values : NULL, scounts, displs, MPI_DOUBLE, 
                 local_csr.values, local_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(my_rank==0 ? global_csr.col_ind : NULL, scounts, displs, MPI_INT, 
                 local_csr.col_ind, local_nnz, MPI_INT, 0, MPI_COMM_WORLD);
    
    // 4. Διανομή row_ptr (Χρήση απλού Scatter καθώς κάθε διεργασία έχει ίδιο αριθμό γραμμών)
    MPI_Scatter(my_rank==0 ? global_csr.row_ptr : NULL, local_n, MPI_INT,
                local_csr.row_ptr, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    // 5. Normalization: Διόρθωση των δεικτών row_ptr ώστε να είναι τοπικοί (0-based)
    int start_idx = local_csr.row_ptr[0];
    for(int i=0; i<local_n; i++) local_csr.row_ptr[i] -= start_idx;
    local_csr.row_ptr[local_n] = local_nnz; // Τελευταίο στοιχείο = συνολικά στοιχεία

    MPI_Barrier(MPI_COMM_WORLD);
    GET_TIME(t_csr_comm_end);

    // 6. Κύριος Βρόχος Υπολογισμού CSR (SpMV Kernel)
    double *local_y = calloc(local_n, sizeof(double)); // Τοπικό αποτέλεσμα

    MPI_Barrier(MPI_COMM_WORLD);
    GET_TIME(t_csr_calc_start);

    for (int iter = 0; iter < iters; iter++) {
        // Υπολογισμός y = A * x (μόνο για τα μη-μηδενικά)
        for (int i = 0; i < local_n; i++) {
            double sum = 0.0;
            // Διασχίζουμε μόνο τα στοιχεία που υπάρχουν (αποδοτικότητα CSR)
            for (int j = local_csr.row_ptr[i]; j < local_csr.row_ptr[i+1]; j++) {
                sum += local_csr.values[j] * x[local_csr.col_ind[j]];
            }
            local_y[i] = sum;
        }
        // Συλλογή αποτελεσμάτων και ανανέωση του x για την επόμενη επανάληψη
        MPI_Allgather(local_y, local_n, MPI_DOUBLE, x, local_n, MPI_DOUBLE, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    GET_TIME(t_csr_calc_end);

    // Καθαρισμός CSR πινάκων πριν το Dense πείραμα
    free(local_csr.values); free(local_csr.col_ind); free(local_csr.row_ptr);

    /* ======================================================
       PHASE 3: DENSE PARALLEL DISTRIBUTION & CALCULATION
       ====================================================== */
    
    // Επαναφορά του x στην αρχική κατάσταση
    memcpy(x, x_copy, n * sizeof(double));

    // Δέσμευση τοπικού πίνακα Dense (local_n * N στοιχεία)
    double *local_A_dense = malloc(local_n * n * sizeof(double));

    MPI_Barrier(MPI_COMM_WORLD);
    GET_TIME(t_dense_comm_start);

    // Διανομή Dense πίνακα (Χρήση απλού Scatter λόγω σταθερού μεγέθους)
    MPI_Scatter(A_dense_global, local_n * n, MPI_DOUBLE,
                local_A_dense, local_n * n, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    GET_TIME(t_dense_comm_end);

    // Dense Loop
    MPI_Barrier(MPI_COMM_WORLD);
    GET_TIME(t_dense_calc_start);

    for (int iter = 0; iter < iters; iter++) {
        // Υπολογισμός Dense (Πράξεις και με τα μηδενικά)
        for (int i = 0; i < local_n; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                sum += local_A_dense[i*n + j] * x[j];
            }
            local_y[i] = sum;
        }
        // Συγχρονισμός αποτελεσμάτων
        MPI_Allgather(local_y, local_n, MPI_DOUBLE, x, local_n, MPI_DOUBLE, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    GET_TIME(t_dense_calc_end);

    /* ======================================================
       PHASE 4: RESULTS & CLEANUP
       ====================================================== */
    
    // Συλλογή τελικού αποτελέσματος (μόνο για επιβεβαίωση στον Master)
    double *final_result = NULL;
    if (my_rank == 0) final_result = malloc(n * sizeof(double));
    if (my_rank == 0) memcpy(final_result, x, n * sizeof(double));

    if (my_rank == 0) {
        // Υπολογισμός συνολικών χρόνων
        double csr_total = (t_csr_create_end - t_csr_create_start) + 
                           (t_csr_comm_end - t_csr_comm_start) +     
                           (t_csr_calc_end - t_csr_calc_start);      
        
        double dense_total = (t_dense_comm_end - t_dense_comm_start) + 
                             (t_dense_calc_end - t_dense_calc_start);

        // Εκτύπωση αποτελεσμάτων σύμφωνα με την εκφώνηση
        printf("\n=== RESULTS (N=%d, Sparsity=%.2f, P=%d, Iters=%d) ===\n", n, sparsity, comm_sz, iters);
        printf("(i)   CSR Creation Time:      %e sec\n", t_csr_create_end - t_csr_create_start);
        printf("(ii)  CSR Comm Time (Distr):  %e sec\n", t_csr_comm_end - t_csr_comm_start);
        printf("(iii) CSR Calc Time:          %e sec\n", t_csr_calc_end - t_csr_calc_start);
        printf("(iv)  Total CSR Time:         %e sec\n", csr_total);
        printf("(v)   Total Dense Time (MPI): %e sec\n", dense_total);
        printf("----------------------------------------------------\n");
        printf("Dense Comm Time:              %e sec\n", t_dense_comm_end - t_dense_comm_start);
        printf("Dense Calc Time:              %e sec\n", t_dense_calc_end - t_dense_calc_start);

        if (n <= 10) {
            printf("Final Result Vector: ");
            for(int k=0; k<n; k++) printf("%.1f ", final_result[k]);
            printf("\n");
        }
        
        // Αποδέσμευση μνήμης Master
        free(A_dense_global); free(x_global); free(final_result); free_csr(&global_csr);
        free(scounts); free(displs);
    }

    // Αποδέσμευση τοπικής μνήμης
    free(x); free(x_copy); free(local_y); free(local_A_dense);
    MPI_Finalize();
    return 0;
}