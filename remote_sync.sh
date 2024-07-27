#!/bin/bash

# Check if a server name was provided
if [ -z "$1" ]; then
  echo "Usage: $0 server_name"
  exit 1
fi

# Define your server names and associated source and destination paths
case "$1" in
  "lightning")
    REMOTE_USER="s_01j3e62793qd2qhydg1p069sjd"
    REMOTE_HOST="ssh.lightning.ai"
    SRC_PATH="/teamspace/studios/this_studio/ArcSolver/runs"
    DEST_PATH="/Users/abhishekaggarwal/synced_repos/ArcSolver/lightning_runs"
    ;;
  "lambda")
    REMOTE_USER="user_server2"
    REMOTE_HOST="server2.example.com"
    SRC_PATH="/path/to/server2/source"
    DEST_PATH="/path/to/local/server2/destination"
    ;;
  # Add more servers here
  *)
    echo "Unknown server: $1"
    exit 1
    ;;
esac

# Trap Ctrl+C (SIGINT) to break the loop and exit
trap "echo 'Sync interrupted. Exiting.'; exit" SIGINT

# Perform the rsync operation in an infinite loop
while true; do
  rsync -auz --progress "${REMOTE_USER}@${REMOTE_HOST}:${SRC_PATH}" "${DEST_PATH}"
  echo "Sync complete. Waiting for 30 seconds before the next sync..."
  sleep 5
done
