#!/bin/bash

SESSION_NAME="ivan-parallel-v5-eu"

# Check if the session already exists
tmux has-session -t $SESSION_NAME 2>/dev/null

# If the session doesn't exist, create it
if [ $? != 0 ]; then
  tmux new-session -d -s $SESSION_NAME
fi

# Function to set up a window
setup_window() {
    local window_number=$1
    tmux new-window -t $SESSION_NAME:$window_number
    tmux send-keys -t $TMUX_SESSION_NAME:$window_number 'conda activate ivan' Enter
    tmux send-keys -t $TMUX_SESSION_NAME:$window_number 'sleep 2' Enter  # Give conda time to activate
    tmux send-keys -t $TMUX_SESSION_NAME:$window_number './launch_scripts/1b_llama_v5_eu.sh' Enter
}

# Set up N windows
for i in {0..16}
do
    setup_window $i
    echo "Window $i set up"
done

# Remove the first window that was created by default
tmux kill-window -t $SESSION_NAME:0

# Select the first window
tmux select-window -t $SESSION_NAME:1

echo "Attach to the session with: tmux attach-session -t $SESSION_NAME"
