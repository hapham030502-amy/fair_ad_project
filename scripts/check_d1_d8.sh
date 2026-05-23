#!/usr/bin/env bash
set -e

echo "[CHECK 1] Generate D1, D2, D5"
python -m src.generate_d1_d2_d5

echo "[CHECK 2] Check data loader - Adult"
python -m src.data_loader --dataset adult

echo "[CHECK 3] Check data loader - Credit Default"
python -m src.data_loader --dataset credit_default

echo "[CHECK 4] Check metrics unit tests"
python -m unittest tests/test_metrics.py

echo "[OK] D1-D5 và D3 technical checks đạt."
