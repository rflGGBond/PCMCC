#!/bin/bash

# Set the project root directory
PROJECT_ROOT=$(pwd)

# Add the project root to PYTHONPATH so Python can find the mapcmcc package
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

echo "Starting MAPCMCC..."
echo "Project Root: $PROJECT_ROOT"

# Check if required packages are installed (optional, basic check)
python -c "import networkx" 2>/dev/null || { echo "Error: networkx is not installed."; exit 1; }
python -c "import igraph" 2>/dev/null || { echo "Error: igraph is not installed."; exit 1; }
python -c "import leidenalg" 2>/dev/null || { echo "Error: leidenalg is not installed."; exit 1; }

# Run the main script
python mapcmcc/main.py
