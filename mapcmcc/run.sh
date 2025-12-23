#!/bin/bash

# Configuration Parameters
GRAPH_NAME="facebook"
TOTAL_BUDGET=50
NUM_COMMUNITIES=4
MAX_GEN=20
T_COMM=5

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Running MAPCMCC with:"
echo "GRAPH_NAME=$GRAPH_NAME"
echo "TOTAL_BUDGET=$TOTAL_BUDGET"
echo "NUM_COMMUNITIES=$NUM_COMMUNITIES"
echo "MAX_GEN=$MAX_GEN"
echo "T_COMM=$T_COMM"
echo "-----------------------------------"

# Run the python script
# Ensure you are in the correct environment or have requirements installed
python3 mapcmcc/run.py \
    --graph_name "$GRAPH_NAME" \
    --total_budget "$TOTAL_BUDGET" \
    --num_communities "$NUM_COMMUNITIES" \
    --max_gen "$MAX_GEN" \
    --t_comm "$T_COMM"
