# WSL2 + Devcontainer Port Forwarding for External Access

**Tags:** `wsl2` `devcontainer` `port-forwarding` `tailscale` `firewall` `networking` `webhook` `troubleshooting`

## Problem

When running forge-bot inside a VS Code devcontainer on WSL2, external machines (e.g. a Gitea server on Tailscale) cannot reach the webhook endpoint on port 8080, even though `uvicorn` is listening on `0.0.0.0:8080` inside the container.

## Root Cause

The traffic path is: **Gitea → Tailscale → Windows host → WSL2 → devcontainer**

Two things block this by default:

1. **VS Code port forwarding binds to `127.0.0.1`** on the Windows host, so only localhost connections work — external traffic (including Tailscale) is rejected.
2. **Windows Firewall** blocks inbound connections on non-standard ports.

## Solution

### Step 1: Windows Firewall — allow port 8080

Run in **PowerShell as Administrator**:

```powershell
New-NetFirewallRule -DisplayName "forge-bot webhook" -Direction Inbound -LocalPort 8080 -Protocol TCP -Action Allow
```

To remove later:

```powershell
Remove-NetFirewallRule -DisplayName "forge-bot webhook"
```

### Step 2: Port proxy — bridge external traffic to localhost

VS Code's forwarder listens on `127.0.0.1:8080`, but external traffic arrives on the machine's real interfaces. Use `netsh` to bridge them.

Run in **PowerShell as Administrator**:

```powershell
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=8080 connectaddress=127.0.0.1 connectport=8080
```

This forwards any traffic arriving on `0.0.0.0:8080` to `127.0.0.1:8080` where VS Code's port forwarder is listening.

To remove later:

```powershell
netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=8080
```

To list all active proxies:

```powershell
netsh interface portproxy show all
```

### Step 3: Verify

From the external machine (e.g. Gitea server):

```bash
curl http://<your-tailscale-hostname>:8080/health
# Expected: {"status":"ok"}
```

## Things That Don't Work

| Approach | Why it fails |
|---|---|
| `remote.localPortHost: allInterfaces` in VS Code settings | Doesn't reliably apply to devcontainer-forwarded ports on WSL2 |
| `portsAttributes` with `"onAutoForward": "silent"` in devcontainer.json | Controls auto-forward behavior, not bind address |
| Port Visibility → Public (right-click in Ports tab) | Only available in GitHub Codespaces, not local devcontainers |

## Notes

- The `netsh portproxy` rule persists across reboots. Remove it when no longer needed.
- The firewall rule also persists. Scope it to the Tailscale interface if you want tighter security:
  ```powershell
  New-NetFirewallRule -DisplayName "forge-bot webhook" -Direction Inbound -LocalPort 8080 -Protocol TCP -Action Allow -InterfaceAlias "Tailscale"
  ```
- If you change the webhook port, update both the firewall rule and the portproxy rule accordingly.
