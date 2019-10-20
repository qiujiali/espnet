MAIN_ROOT=$PWD/../../..
source $MAIN_ROOT/../venv/etc/profile.d/conda.sh && conda deactivate && conda activate
export PATH=$MAIN_ROOT/utils:$MAIN_ROOT/espnet/bin:$PWD/utils/:$PATH
export PYTHONPATH=$MAIN_ROOT:$PYTHONPATH
export OMP_NUM_THREADS=1
