#!/bin/bash
cd /mnt/e/Python_project/bybit_grid_bot
source venv/bin/activate
export PYTHONPATH=/mnt/e/Python_project/bybit_grid_bot
streamlit run web_ui/app.py
