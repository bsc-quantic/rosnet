#!/bin/bash

PARAMS=""
NUM_NODES=1
EXEC_TIME=2
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
		-*=*|--*=*)
			echo "Error: Unsupported ="
			exit 1
			;;
		*)
			PARAMS="$PARAMS $1"
			shift 1
			;;
	esac
done

export PYTHONPATH=$(pwd):$PYTHONPATH
enqueue_compss --num_nodes=$NUM_NODES --qos=debug --log_level=info --exec_time=$EXEC_TIME --summary --graph $PARAMS
