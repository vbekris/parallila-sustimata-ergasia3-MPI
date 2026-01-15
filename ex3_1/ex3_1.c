#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "timer.h" 

int main(int argc, char* argv[]) {
    int my_rank, comm_sz;
    int n, N, local_n;
    
    // Αρχικοποίηση περιβάλλοντος MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    // Έλεγχος ορισμάτων εισόδου
    if (argc != 2) {
        if (my_rank == 0) printf("Usage: %s <degree n>\n", argv[0]);
        MPI_Finalize();
        return 0;
    }

    n = atoi(argv[1]);
    N = n + 1; // Πλήθος συντελεστών (βαθμός n -> n+1 όροι)

    // Έλεγχος διαιρετότητας: Το N πρέπει να διαιρείται ακριβώς με το πλήθος διεργασιών
    // για να λειτουργήσει σωστά η MPI_Scatter.
    if (N % comm_sz != 0) {
        if (my_rank == 0) {
            printf("Error: Matrix size N=%d is not divisible by P=%d.\n", N, comm_sz);
        }
        MPI_Finalize();
        return 0;
    }

    // Υπολογισμός του μεγέθους του πίνακα A που αναλογεί σε κάθε διεργασία
    local_n = N / comm_sz;

    /* --- Δέσμευση Μνήμης --- */
    
    // 1. local_A: Αποθηκεύει το τμήμα του πίνακα A που θα επεξεργαστεί η διεργασία
    int *local_A = (int*) malloc(local_n * sizeof(int));
    
    // 2. B: Αποθηκεύει ολόκληρο το πολυώνυμο B.
    // Χρειαζόμαστε όλο το B σε κάθε διεργασία για να γίνει σωστά η συνέλιξη (convolution).
    int *B = (int*) malloc(N * sizeof(int));

    // 3. local_C: Πίνακας για τα μερικά αποτελέσματα.
    // Το γινόμενο πολυωνύμων βαθμού n έχει βαθμό 2n, άρα μέγεθος 2n+1.
    int res_size = 2 * n + 1;
    // Χρήση calloc για αρχικοποίηση με 0, ώστε να κάνουμε += αργότερα.
    int *local_C = (int*) calloc(res_size, sizeof(int)); 

    // Δείκτες για τους Global πίνακες (χρήση μόνο από τον Master)
    int *A = NULL;       
    int *final_C = NULL; 
    
    // Ο Master δεσμεύει και αρχικοποιεί τα δεδομένα
    if (my_rank == 0) {
        A = (int*) malloc(N * sizeof(int));
        final_C = (int*) malloc(res_size * sizeof(int));

        printf("Master: Initializing polynomials (Degree n=%d, Coeffs N=%d)...\n", n, N);
        srand(42); // Seed για επαναληψιμότητα των πειραμάτων
        for (int i = 0; i < N; i++) {
            A[i] = (rand() % 10) + 1; 
            B[i] = (rand() % 10) + 1; 
        }
    }

    /* --- Φάση Επικοινωνίας (Distribution) --- */
    
    double t_start, t_comm_end;

    // Barrier για να ξεκινήσει η χρονομέτρηση ταυτόχρονα
    MPI_Barrier(MPI_COMM_WORLD);
    GET_TIME(t_start);

    // Διανομή του πίνακα A: Κάθε διεργασία παίρνει local_n στοιχεία
    MPI_Scatter(A, local_n, MPI_INT, 
                local_A, local_n, MPI_INT, 
                0, MPI_COMM_WORLD);

    // Broadcast του πίνακα B: Όλες οι διεργασίες λαμβάνουν όλο το B
    MPI_Bcast(B, N, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD); 
    GET_TIME(t_comm_end);

    // Debug print για επιβεβαίωση λήψης δεδομένων
    printf("Rank %d received local_A[0]=%d and B[0]=%d\n", my_rank, local_A[0], B[0]);
    

    /* --- Φάση Υπολογισμού (Convolution Kernel) --- */
    
    double t_calc_end;
    
    // Υπολογισμός του global index από όπου ξεκινάει το local_A της διεργασίας
    int global_offset = my_rank * local_n;

    // Διπλός βρόχος για τον υπολογισμό του γινομένου (Συνέλιξη)
    for (int i = 0; i < local_n; i++) {
        for (int j = 0; j < N; j++) {
            
            // Ο τρέχων όρος του A αντιστοιχεί στη δύναμη x^(global_offset + i)
            int global_i = global_offset + i;
            
            // Οι δυνάμεις προστίθενται στον πολλαπλασιασμό: x^a * x^b = x^(a+b)
            int c_index = global_i + j;
            
            // Προσθήκη στο αντίστοιχο κελί του αποτελέσματος
            local_C[c_index] += local_A[i] * B[j];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD); 
    GET_TIME(t_calc_end);

    // Debug print για έλεγχο υπολογισμού
    printf("Rank %d calculated partial C[%d] = %d\n", 
           my_rank, global_offset, local_C[global_offset]);
           

    /* --- Φάση Συλλογής & Αποτελέσματα (Reduction) --- */

    double t_reduce_end;

    // Συγκέντρωση (Reduction) των μερικών πινάκων local_C στον final_C του Master.
    // Χρησιμοποιούμε MPI_SUM για να αθροίσουμε τους συντελεστές που αντιστοιχούν στην ίδια δύναμη.
    MPI_Reduce(local_C, final_C, res_size, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    GET_TIME(t_reduce_end);

    // Εκτύπωση αποτελεσμάτων από τον Master
    if (my_rank == 0) {
        printf("\n--- RESULTS ---\n");
        printf("Polynomial Degree n: %d (Coeffs N=%d)\n", n, N);
        printf("MPI Processes P:     %d\n", comm_sz);
        printf("--------------------------------------\n");
        printf("(i)   Comm Time (Scatter/Bcast): %e sec\n", t_comm_end - t_start);
        printf("(ii)  Calc Time:                 %e sec\n", t_calc_end - t_comm_end);
        printf("(iii) Reduce Time:               %e sec\n", t_reduce_end - t_calc_end);
        printf("(iv)  Total Parallel Time:       %e sec\n", t_reduce_end - t_start);
        printf("--------------------------------------\n");

        // Εκτύπωση διανύσματος μόνο αν είναι μικρό (για επαλήθευση)
        if (res_size <= 30) {
            printf("Final Result C (Degree 2n): \n");
            for (int i = 0; i < res_size; i++) {
                printf("%d ", final_C[i]);
            }
            printf("\n");
        }
    }

    // Αποδέσμευση μνήμης
    free(local_A);
    free(B);
    free(local_C);

    if (my_rank == 0) {
        free(A);
        free(final_C);
    }

    MPI_Finalize();
    return 0;
}