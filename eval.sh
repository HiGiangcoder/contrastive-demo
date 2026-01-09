#!/bin/bash
set -e

echo "Evaluating all experiments"
echo "=========================="

python evaluate.py

echo "=========================="
echo "Evaluation done."
