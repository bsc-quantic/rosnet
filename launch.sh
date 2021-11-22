#!/bin/bash

# defaults
NUM_NODES=1
USER=$(id -u -n)
GROUP=$(id -g -n)
BASE_LOG_DIR=/gpfs/scratch/$GROUP/$USER
WORKER_WD=/gpfs/scratch/$GROUP/$USER
MASTER_WD=/gpfs/scratch/$GROUP/$USER
QOS=debug

FLAGS=()
PARAMS=()

# parse arguments
while (( "$#" )); do
	case "$1" in
		-t|--exec_time)
			FLAGS+=(--exec_time=$2)
			shift 2
			;;
		-n|--num_nodes)
			NUM_NODES=$2
			shift 2
			;;
		--log_level)
			FLAGS+=(--log_level=$2)
			shift 2
			;;
		--trace)
			FLAGS+=(--tracing=$2)
			shift 2
			;;
		--base_log_dir)
				BASE_LOG_DIR=$2
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
			FLAGS+=(--worker_in_master_cpus=$2)
			shift 2
			;;
		--agents)
			FLAGS+=(--agents=$2)
			shift 2
			;;
		--scheduler)
			FLAGS+=(--scheduler=$2)
			shift 2
			;;
		--gpus_per_node)
			FLAGS+=(--gpus_per_node=$2)
			shift 2
			;;
		--graph)
			FLAGS+=(--graph)
			shift 1
			;;
		--pyenv)
			FLAGS+=(--python_interpreter=$(pyenv which python))
			shift 1
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

enqueue_compss --num_nodes=$NUM_NODES --qos=$QOS --base_log_dir=$BASE_LOG_DIR --worker_working_dir=$WORKER_WD --master_working_dir=$MASTER_WD --pythonpath=$PWD --summary ${FLAGS[*]} ${PARAMS[*]}
