#!/bin/bash

# defaults
PARAMS=()
NUM_NODES=1
EXEC_TIME=2
LOG_LEVEL=off
TRACE=false
USER=$(id -u -n)
GROUP=$(id -g -n)
WORKER_WD=/gpfs/scratch/$GROUP/$USER
MASTER_WD=/gpfs/scratch/$GROUP/$USER
QOS=debug
MASTER_CPUS=0

# parse arguments
while (( "$#" )); do
	case "$1" in
		-t|--exec_time)
			EXEC_TIME=$2
			shift 2
			;;
		-n|--num_nodes)
			NUM_NODES=$2
			shift 2
			;;
		--log_level)
			LOG_LEVEL=$2
			shift 2
			;;
		--trace)
			TRACE=$2
			shift 2
			;;
		--worker_working_dir)
			WORKER_WD=$2
			shift 2
			;;
		--master_working_dir)
			MASTER_WD=$2
			shift 2
			;;
		--qos)
			QOS=$2
			shift 2
			;;
		--master_cpus)
			MASTER_CPUS=$2
			shift 2
			;;
		-*=*|--*=*)
			echo "Error: Unsupported '=' syntax"
			exit 1
			;;
		*)
			PARAMS+=($1)
			shift 1
			;;
	esac
done

# fix execution from working dir
if [[ "${PARAMS[0]}" != /* ]]; then
	PARAMS[0]=$PWD/${PARAMS[0]}
fi

enqueue_compss --num_nodes=$NUM_NODES --qos=$QOS --log_level=$LOG_LEVEL --exec_time=$EXEC_TIME --worker_working_dir=$WORKER_WD --master_working_dir=$MASTER_WD --pythonpath=$PWD --summary --graph --tracing=$TRACE --worker_in_master_cpus=$MASTER_CPUS ${PARAMS[*]}
