#!/bin/bash
for i in {1..2}; do
  echo -e "\nROUND $i\n"
  for j in {1..4}; do
    python run_q_learning_agent.py &
    sleep 2
  done
done