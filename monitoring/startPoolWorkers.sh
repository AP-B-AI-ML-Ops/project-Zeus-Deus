#!/bin/bash
sleep 20
prefect work-pool create --type process monitoring --overwrite
prefect worker start -p monitoring &

python /monitoring.py