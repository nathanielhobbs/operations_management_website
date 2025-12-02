#!/usr/bin/env bash
cd /home/nhobbs/operations_management_website

/home/nhobbs/miniconda3/envs/lp/bin/streamlit run 3d_graph.py \
  --server.port 6072 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false

