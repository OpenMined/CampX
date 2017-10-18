#!/bin/bash
for i in {1..2}; do
  for j in {1..4}; do
    python run_sarsa_agent.py &
    sleep 2
  done
done