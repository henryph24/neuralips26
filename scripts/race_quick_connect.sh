#!/usr/bin/env bash
# Quick-connect to RACE VM — run the MOMENT the professor activates the SSH key.
# Usage: bash scripts/race_quick_connect.sh

set -e
KEY="hungphanphd.pem"
HOST="ec2-user@ec2-13-238-161-176.ap-southeast-2.compute.amazonaws.com"

# Ensure key permissions
chmod 400 "$KEY" 2>/dev/null || true

echo "Connecting to RACE VM..."
ssh -i "$KEY" \
    -o ConnectTimeout=15 \
    -o StrictHostKeyChecking=no \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    "$HOST"
