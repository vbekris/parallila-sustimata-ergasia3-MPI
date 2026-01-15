#!/bin/bash

# --- Ρυθμίσεις Πειραμάτων ---

# Βαθμοί Πολυωνύμων (Degrees n)
# Επιλέγουμε τιμές ώστε το N = n+1 να διαιρείται ΑΚΡΙΒΩΣ με το 32.
# N = 3200   (Πολύ μικρό - Κυριαρχεί η επικοινωνία) -> n = 3199
# N = 32000  (Μικρό) -> n = 31999
# N = 102400 (Μεσαίο - Τυπικό) -> n = 102399
# N = 204800 (Μεγάλο - Κυριαρχεί ο υπολογισμός) -> n = 204799
DEGREES="3199 31999 102399 204799"

# Αριθμός Διεργασιών (P)
PROCESSES="1 2 4 8 16 32"

OUTPUT_FILE="results_ex3_1_graph_data.txt"
MACHINES_FILE="machines"

# --- Compile ---
echo "--- Compiling Project ---"
make clean
make

if [ ! -f ./ex3_1 ]; then
    echo "❌ Error: Compilation failed!"
    exit 1
fi

# --- Header ---
echo "==================================================================" > $OUTPUT_FILE
echo " EXPERIMENT 3.1 DATA COLLECTION" >> $OUTPUT_FILE
echo " Degrees: $DEGREES" >> $OUTPUT_FILE
echo " Date: $(date)" >> $OUTPUT_FILE
echo "==================================================================" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# --- Loops ---
echo "🚀 Starting Experiments ..."

for n in $DEGREES; do
    N=$((n+1))
    echo "------------------------------------------------------------------" >> $OUTPUT_FILE
    echo ">>> POLYNOMIAL DEGREE n = $n (Size N=$N) <<<" >> $OUTPUT_FILE
    echo "------------------------------------------------------------------" >> $OUTPUT_FILE
    
    for p in $PROCESSES; do
        # Safety Check: Διαιρετότητα
        if (( N % p != 0 )); then
            continue
        fi

        echo "   Running: n=$n | P=$p"
        echo "   --- Processes: P=$p ---" >> $OUTPUT_FILE

        # Τρέχουμε ΜΙΑ φορά για ταχύτητα
        mpiexec -f $MACHINES_FILE -n $p ./ex3_1 $n >> $OUTPUT_FILE
        
        echo "   ---------------------" >> $OUTPUT_FILE
    done
done

echo "✅ All experiments finished!"
echo "📄 Results saved in: $OUTPUT_FILE"