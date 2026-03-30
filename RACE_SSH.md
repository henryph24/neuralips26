# RACE VM SSH Access

## Connection command
```bash
ssh -i hungphanphd.pem ec2-user@ec2-13-238-161-176.ap-southeast-2.compute.amazonaws.com
```

- **Key**: `hungphanphd.pem` (in repo root)
- **User**: `ec2-user`
- **Host**: `ec2-13-238-161-176.ap-southeast-2.compute.amazonaws.com`
- **GPU**: NVIDIA A10G (23GB VRAM), CUDA 12.4
- **OS**: Ubuntu 22.04 on AWS (kernel 6.8.0)

## IP whitelisting

SSH access requires your public IP in the RACE security group (port 22).

To update:
1. Get your current IP: `curl ifconfig.me`
2. Professor logs into https://race.rmit.edu.au
3. Workspaces → select workspace → **Edit Security Group**
4. Add `<your-ip>/32` on **port 22**

## Notes
- Key permissions must be `chmod 400` (`chmod 400 hungphanphd.pem`)
- The 60-second countdown after clicking "Use this SSH Key" in RACE applies when authorizing keys via the web UI
