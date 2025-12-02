#!/usr/bin/env bash
set -e

cd /home/nhobbs/operations_management_website

# Update code
git pull

# Restart the Streamlit service
sudo systemctl restart operations_streamlit.service

