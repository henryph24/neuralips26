#!/usr/bin/env bash
# Whitelist an IP at the machine level (iptables + ufw if present).
# Run ON the RACE VM after connecting via SSH.
#
# Usage: sudo bash race_whitelist_ip.sh [IP]
#   Default IP: 23.27.187.5

set -euo pipefail

IP="${1:-23.27.187.5}"
echo "=== Whitelisting $IP for SSH (port 22) at machine level ==="

# ---- iptables (works on all Linux) ----
# Check if rule already exists to be idempotent
if iptables -C INPUT -s "$IP/32" -p tcp --dport 22 -j ACCEPT 2>/dev/null; then
    echo "[iptables] Rule already exists for $IP — skipping."
else
    # Insert at top of INPUT chain so it's evaluated before any DROP/REJECT
    iptables -I INPUT 1 -s "$IP/32" -p tcp --dport 22 -j ACCEPT
    echo "[iptables] Added ACCEPT rule for $IP:22"
fi

# Persist iptables rules across reboots
if command -v netfilter-persistent &>/dev/null; then
    netfilter-persistent save
    echo "[iptables] Rules persisted via netfilter-persistent."
elif command -v iptables-save &>/dev/null; then
    mkdir -p /etc/iptables
    iptables-save > /etc/iptables/rules.v4
    echo "[iptables] Rules saved to /etc/iptables/rules.v4"
fi

# ---- ufw (if installed and active) ----
if command -v ufw &>/dev/null; then
    UFW_STATUS=$(ufw status | head -1)
    if echo "$UFW_STATUS" | grep -q "active"; then
        ufw allow from "$IP" to any port 22 proto tcp comment "RACE-whitelist"
        echo "[ufw] Rule added for $IP:22"
    else
        echo "[ufw] Inactive — skipping (iptables rule is sufficient)."
    fi
else
    echo "[ufw] Not installed — skipping (iptables rule is sufficient)."
fi

# ---- Verify ----
echo ""
echo "=== Verification ==="
echo "iptables rules matching $IP:"
iptables -L INPUT -n --line-numbers | grep "$IP" || echo "  (none found — check manually)"
echo ""
echo "Current SSH listening:"
ss -tlnp | grep :22 || netstat -tlnp 2>/dev/null | grep :22 || echo "  (could not check)"
echo ""
echo "Done. $IP is now whitelisted at the machine level."
echo "NOTE: This does NOT bypass the AWS security group — the professor"
echo "must also whitelist $IP/32 on port 22 in the RACE security group"
echo "for external access to reach this machine."
