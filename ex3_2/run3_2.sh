#!/bin/bash

# --- Παράμετροι Πειραμάτων ---

# 1. Διαστάσεις Πίνακα (10^3 έως 10^4)
# Τιμές που διαιρούνται με το 32 (P) για ομαλή κατανομή
SIZES="1024 10240" 

# 2. Ποσοστά Μηδενικών (Sparsity)
# 0.00 = Dense, 0.50 = Mixed, 0.99 = Sparse (CSR Target)
SPARSITIES="0.00 0.50 0.99"

# 3. Αριθμός Επαναλήψεων (Variable Iterations)
ITER_COUNTS="1 10 20"

# 4. Αριθμός Διεργασιών (Scaling)
PROCESSES="1 2 4 8 16 32"

OUTPUT_FILE="results_final_report.txt"
MACHINES_FILE="machines"

# --- Build ---
echo "--- Compiling Project ---"
make clean
make

if [ ! -f ./ex3_2 ]; then
    echo "❌ Error: Compilation failed!"
    exit 1
fi

# --- Output Initialization ---
echo "==================================================================" > $OUTPUT_FILE
echo " FINAL REPORT EXPERIMENTS (Ex 3.2)" >> $OUTPUT_FILE
echo " Sizes: 10^3 to 10^4 | Iters: 1 to 20 | Sparsity: 0% to 99%" >> $OUTPUT_FILE
echo " Date: $(date)" >> $OUTPUT_FILE
echo "==================================================================" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# --- Execution Loops ---
echo "🚀 Starting Experiments..."

for n in $SIZES; do
    echo "------------------------------------------------------------------" >> $OUTPUT_FILE
    echo ">>> MATRIX SIZE N = $n <<<" >> $OUTPUT_FILE
    echo "------------------------------------------------------------------" >> $OUTPUT_FILE
    
    for sp in $SPARSITIES; do
        for iters in $ITER_COUNTS; do
            
            echo "" >> $OUTPUT_FILE
            echo "   [Sparsity: $sp | Iterations: $iters]" >> $OUTPUT_FILE
            
            for p in $PROCESSES; do
                # Έλεγχος συμβατότητας N και P
                if (( n % p != 0 )); then
                    continue
                fi

                echo "   Running: N=$n | Sparsity=$sp | Iters=$iters | P=$p"
                
                echo "   --- Processes: P=$p ---" >> $OUTPUT_FILE
                
                mpiexec -f $MACHINES_FILE -n $p ./ex3_2 $n $sp $iters >> $OUTPUT_FILE
                
                echo "   ---------------------" >> $OUTPUT_FILE
            done
        done
    done
done

echo "✅ All experiments finished!"
echo "📄 Results saved in: $OUTPUT_FILE"