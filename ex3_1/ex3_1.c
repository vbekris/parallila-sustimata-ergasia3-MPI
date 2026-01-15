#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "timer.h" // Το αρχείο που ανέβασες

int main(int argc, char* argv[]) {
    int my_rank, comm_sz;
    int n, N, local_n;
    
    // --- MPI Init ---
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    // --- Έλεγχος Ορισμάτων (Μόνο ο 0 μιλάει) ---
    if (argc != 2) {
        if (my_rank == 0) printf("Usage: %s <degree n>\n", argv[0]);
        MPI_Finalize();
        return 0;
    }

    n = atoi(argv[1]);
    N = n + 1; // Οι συντελεστές είναι n+1 (π.χ. βαθμός 1 -> ax+b -> 2 όροι)

    // --- Safety Check: Διαιρετότητα ---
    if (N % comm_sz != 0) {
        if (my_rank == 0) {
            printf("Error: Matrix size N=%d is not divisible by P=%d.\n", N, comm_sz);
            printf("Please choose different parameters.\n");
        }
        MPI_Finalize();
        return 0;
    }

    local_n = N / comm_sz; // Πόσα στοιχεία του A θα πάρει ο καθένας

    // --- Δέσμευση Μνήμης (Allocation) ---
    
    // 1. local_A: Το κομμάτι του A που θα επεξεργαστώ
    int *local_A = (int*) malloc(local_n * sizeof(int));
    
    // 2. B: Ολόκληρο το B (το χρειάζονται όλοι για τον πολλαπλασιασμό)
    int *B = (int*) malloc(N * sizeof(int));

    // 3. local_C: Το μερικό αποτέλεσμα.
    // Προσοχή: Το αποτέλεσμα πολ/σμου βαθμού n έχει βαθμό 2n.
    // Μέγεθος αποτελέσματος = (2n + 1).
    int res_size = 2 * n + 1;
    int *local_C = (int*) calloc(res_size, sizeof(int)); // calloc = γέμισμα με 0

    // --- Global Arrays (ΜΟΝΟ στον Master) ---
    int *A = NULL;       // Ο αρχικός πίνακας A
    int *final_C = NULL; // Ο τελικός πίνακας αποτελεσμάτων
    
    if (my_rank == 0) {
        A = (int*) malloc(N * sizeof(int));
        final_C = (int*) malloc(res_size * sizeof(int));

        // Γέμισμα με τυχαία δεδομένα
        printf("Master: Initializing polynomials (Degree n=%d, Coeffs N=%d)...\n", n, N);
        srand(42); // Σταθερό seed για να βγάζει τα ίδια κάθε φορά (debugging)
        for (int i = 0; i < N; i++) {
            A[i] = (rand() % 10) + 1; // Τιμές 1-10
            B[i] = (rand() % 10) + 1; // Γεμίζουμε το B του Master απευθείας
        }
    }

    // --- ΤΕΛΟΣ ΚΟΜΜΑΤΙΟΥ 1 ---
    // Εδώ σταματάμε για τώρα.
    // Δεν έχουμε κάνει ακόμη καμία επικοινωνία.


    // --- ΚΟΜΜΑΤΙ 2: Επικοινωνία & Χρονομέτρηση ---
    
    double t_start, t_comm_end;

    // 1. Συγχρονισμός: Περιμένουμε όλους να φτάσουν εδώ πριν ξεκινήσει το χρονόμετρο
    MPI_Barrier(MPI_COMM_WORLD);
    GET_TIME(t_start);

    // 2. Scatter: Μοιράζουμε το A
    // Ορίσματα: 
    // (SourceBuf, SendCount/Process, Type, DestBuf, RecvCount, Type, Root, Comm)
    // Προσοχή: Το A είναι NULL στους Workers, αλλά το MPI το αγνοεί εκεί.
    MPI_Scatter(A, local_n, MPI_INT, 
                local_A, local_n, MPI_INT, 
                0, MPI_COMM_WORLD);

    // 3. Broadcast: Στέλνουμε το B σε όλους
    // Ορίσματα: (Buffer, Count, Type, Root, Comm)
    // Εδώ στέλνουμε ΟΛΟΚΛΗΡΟ το N, όχι local_n!
    MPI_Bcast(B, N, MPI_INT, 0, MPI_COMM_WORLD);

    // Σταματάμε το χρονόμετρο επικοινωνίας
    MPI_Barrier(MPI_COMM_WORLD); // Προαιρετικό, για ακρίβεια μέτρησης
    GET_TIME(t_comm_end);

    // Debugging Print (Για να δούμε ότι δούλεψε)
    // Τυπώνουμε το πρώτο στοιχείο που έλαβε ο καθένας για επιβεβαίωση
    printf("Rank %d received local_A[0]=%d and B[0]=%d\n", 
            my_rank, local_A[0], B[0]);
    
    // --- ΤΕΛΟΣ ΚΟΜΜΑΤΙΟΥ 2 ---


// --- ΚΟΜΜΑΤΙ 3: Υπολογισμός (Convolution) ---
    
    double t_calc_end;
    
    // 1. Υπολογισμός του Global Offset
    // Π.χ. Αν local_n=3, ο Rank 1 ξεκινάει από το δείκτη 3 του αρχικού A.
    int global_offset = my_rank * local_n;

    // 2. Ο Κύριος Βρόχος Υπολογισμού
    // Για κάθε στοιχείο του ΔΙΚΟΥ ΜΟΥ κομματιού του A...
    for (int i = 0; i < local_n; i++) {
        
        // ...το πολλαπλασιάζω με ΟΛΑ τα στοιχεία του B
        for (int j = 0; j < N; j++) {
            
            // Ποια είναι η πραγματική θέση του A;
            int global_i = global_offset + i;
            
            // Ποια θέση του αποτελέσματος C επηρεάζουμε; (Βαθμός A + Βαθμός B)
            int c_index = global_i + j;
            
            // Προσθήκη στο μερικό άθροισμα
            local_C[c_index] += local_A[i] * B[j];
        }
    }

    // 3. Χρονομέτρηση
    MPI_Barrier(MPI_COMM_WORLD); // Βεβαιωνόμαστε ότι όλοι τελείωσαν τους υπολογισμούς
    GET_TIME(t_calc_end);

    // Debugging Print (Για να δούμε ότι κάτι υπολογίστηκε)
    // Τυπώνουμε ένα τυχαίο στοιχείο του αποτελέσματος για έλεγχο
    // Π.χ. τη θέση [global_offset], που σίγουρα έχει τιμή.
    printf("Rank %d calculated partial C[%d] = %d\n", 
           my_rank, global_offset, local_C[global_offset]);
           
    // --- ΤΕΛΟΣ ΚΟΜΜΑΤΙΟΥ 3 ---


    
// --- ΚΟΜΜΑΤΙ 4: Συλλογή & Τερματισμός ---

    double t_reduce_end;

    // 1. MPI Reduce: Μαζεύουμε όλα τα local_C στο final_C του Master
    // Προσοχή: Το final_C υπάρχει μόνο στον Rank 0. Στους άλλους βάζουμε NULL (ή τίποτα).
    MPI_Reduce(local_C, final_C, res_size, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Σταματάμε το χρονόμετρο για το Reduce
    GET_TIME(t_reduce_end);

    // 2. Εκτύπωση Αποτελεσμάτων (ΜΟΝΟ ο Master)
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

        // Προαιρετικά: Εκτύπωση αποτελέσματος για μικρά N
        if (res_size <= 30) {
            printf("Final Result C (Degree 2n): \n");
            for (int i = 0; i < res_size; i++) {
                printf("%d ", final_C[i]);
            }
            printf("\n");
        }
    }

    // 3. Cleanup (Καθαρισμός Μνήμης)
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