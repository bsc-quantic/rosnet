#!/bin/bash

PARAMS=""
EXEC_TIME=2
while (( "$#" )); do
	case "$1" in
		-t|--exec_time)
			EXEC_TIME=$2
			shift 2
			;;
		-t=*|--exec_time=*)
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
enqueue_compss --num_nodes=1 --qos=debug --log_level=info --exec_time=$EXEC_TIME --summary --graph $PARAMS
