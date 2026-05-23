#!/usr/bin/env bash
set -e

echo "[D3-1] Check Adult loader"
python -m src.data_loader --dataset adult

echo "[D3-2] Check Credit Default loader"
python -m src.data_loader --dataset credit_default

echo "[D3-3] Check metrics unit tests"
python -m unittest tests/test_metrics.py

echo "[OK] D3 đạt yêu cầu kỹ thuật cơ bản."
