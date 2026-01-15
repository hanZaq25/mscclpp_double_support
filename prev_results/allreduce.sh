#!/bin/bash
LOGFILE="allreduce_k0_to_k7_16B_to_1MB.log"
rm -f $LOGFILE

echo "Starting AllReduce sweep for k = 0..7" | tee -a $LOGFILE

for k in {0..7}; do
    echo "----------------------------------------" | tee -a $LOGFILE
    echo "Running k = $k" | tee -a $LOGFILE
    echo "----------------------------------------" | tee -a $LOGFILE
    
    # Run directly (no nohup/wait needed inside the loop)
    # We use 2>&1 to capture both errors and standard output
    mpirun -np 4 \
        --bind-to core \
        --map-by socket \
        ./test/mscclpp-test/allreduce_test_perf \
            -b 16 \
            -e 1024 \
            -f 2 \
            -n 1000 \
            -w 10 \
            -k $k \
        >> $LOGFILE 2>&1
    
    # Check the exit code of the mpirun command immediately
    exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        echo "❌ ERROR: k = $k exited with code $exit_code" | tee -a $LOGFILE
        # Optional: exit the whole script if one fails?
        # exit 1 
    else
        echo "✔ Completed k = $k successfully" | tee -a $LOGFILE
    fi
done

echo "All runs completed." | tee -a $LOGFILE