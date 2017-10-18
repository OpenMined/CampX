#!/bin/bash
for i in {1..20}; do
  python run_q_learning_agent.py &
  sleep 2
done