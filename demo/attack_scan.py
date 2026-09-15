"""Lightweight SYN-scan generator for the live demo.

Sweeps a victim across a range of ports so the response flows (SYN-ACK / RST)
are visible to the capture agent sniffing the same interface.  Expect the
dashboard to show PortScan -> ALERT (or BLOCK on wider/longer sweeps).

Best demo target: your own IP (RSTs come back instantly, no ARP needed):
    .\\venv\\Scripts\\python.exe demo\\attack_scan.py --vip 192.168.110.203 --ports 80
    .\\venv\\Scripts\\python.exe demo\\attack_scan.py --vip 192.168.110.203 --ports 300

Or the router for a broadcast sweep (ARP warnings are harmless):
    .\\venv\\Scripts\\python.exe demo\\attack_scan.py --vip 192.168.110.1 --ports 80

Run from an Administrator PowerShell (raw sockets).
"""

import argparse
import socket
import time


def main(argv=None):
    p = argparse.ArgumentParser(description="SYN sweep for the live IDS demo")
    p.add_argument("--vip", required=True, help="victim IP to sweep")
    p.add_argument("--ports", type=int, default=80,
                   help="number of ports to sweep (default 80)")
    p.add_argument("--pace", type=float, default=0.02,
                   help="seconds between packets (default 0.02)")
    p.add_argument("--sip", default=None,
                   help="spoofed source IP (default: random in your /24)")
    args = p.parse_args(argv)

    from scapy.layers.inet import IP, TCP
    from scapy.sendrecv import send

    try:
        local = socket.gethostbyname(socket.gethostname())
    except OSError:
        local = "192.168.110.203"
    base = ".".join(local.split(".")[:3])

    spoof = args.sip or f"{base}.250"
    victim = socket.gethostbyname(args.vip)

    sent = 0
    try:
        for dport in range(1, args.ports + 1):
            send(IP(src=spoof, dst=victim) /
                 TCP(sport=50000 + sent, dport=dport, flags="S", window=1024),
                 verbose=False)
            sent += 1
            if not sent % 200:
                print(f"  sent {sent} SYNs...")
            time.sleep(args.pace)
    except KeyboardInterrupt:
        pass
    print(f"Sent {sent} SYN packets to {victim} from {spoof}. "
          f"Now check the dashboard telemetry.")


if __name__ == "__main__":
    main()