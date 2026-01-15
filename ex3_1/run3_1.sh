#!/bin/bash

# Ορισμός παραμέτρων
# Χρησιμοποιούμε n=99999 (άρα N=100.000) και n=999999 (άρα N=1.000.000)
# γιατί αυτά τα N διαιρούνται ακριβώς με το 32 (και τις δυνάμεις του 2).
POLYNOMIAL_DEGREES="99999 999999" 
PROCESSES="1 2 4 8 16 32"
OUTPUT_FILE="results.txt"
MACHINES_FILE="machines"

# 1. Compile
echo "--- Compiling ---"
make clean
make

if [ ! -f ./ex3_1 ]; then
    echo "Compilation failed!"
    exit 1
fi

# Καθαρισμός προηγούμενων αποτελεσμάτων
echo "Starting Experiments..." > $OUTPUT_FILE
echo "Date: $(date)" >> $OUTPUT_FILE
echo "--------------------------------" >> $OUTPUT_FILE

# 2. Loops εκτέλεσης
for n in $POLYNOMIAL_DEGREES; do
    N=$((n+1))
    echo "==========================================" >> $OUTPUT_FILE
    echo "Running for Polynomial Degree n=$n (Size N=$N)" >> $OUTPUT_FILE
    echo "==========================================" >> $OUTPUT_FILE
    
    for p in $PROCESSES; do
        echo "Running with P=$p processes..."
        
        # Εκτύπωση επικεφαλίδας στο αρχείο
        echo "--- P = $p ---" >> $OUTPUT_FILE
        
        # ΕΚΤΕΛΕΣΗ MPI
        # -f machines: Χρήση του cluster
        # -n $p: Αριθμός διεργασιών
        mpiexec -f $MACHINES_FILE -n $p ./ex3_1 $n >> $OUTPUT_FILE
        
        echo "Done with P=$p."
    done
done

echo "All experiments finished. Results saved in $OUTPUT_FILE."