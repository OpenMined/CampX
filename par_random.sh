#!/bin/bash
for i in {1..20}; do
  python run_random_agent.py &
  sleep 2
done