ENGINE="integral.tbb"
PUZZLE=${ENGINE%%.*}
SCALES="325 625 1225 2025 3145"
WORKING=.tmp
mkdir ${WORKING}
for SCALE in $SCALES ; do
 
  bin/run_puzzle ${ENGINE} ${SCALE}

done
