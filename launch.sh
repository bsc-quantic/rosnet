#!/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH
enqueue_compss --num_nodes=1 --qos=debug --log_level=info --exec_time=2 --summary --graph $@
