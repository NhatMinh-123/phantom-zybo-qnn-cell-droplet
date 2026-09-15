"""Summarize only camera TCP flows from the scoped Packet Monitor capture."""
from collections import Counter
from pathlib import Path
import json
from scapy.all import PcapNgReader, IP, TCP

root = Path('reports/ethernet_live_20260910/stream_optimization/network_trace')
commands, data_packets = [], []
seen = set()
for packet in PcapNgReader(str(root/'camera.pcapng')):
    if IP not in packet or TCP not in packet:
        continue
    ip, tcp = packet[IP], packet[TCP]
    size = ip.len-ip.ihl*4-tcp.dataofs*4
    if size <= 0:
        continue
    key = (ip.src,tcp.sport,tcp.dport,tcp.seq,size)
    if key in seen:
        continue
    seen.add(key)
    timestamp = float(packet.time)
    if tcp.dport == 7115:
        commands.append((timestamp, bytes(tcp.payload).decode('ascii','replace')))
    if ip.src == '100.100.196.217' and tcp.sport == 7116:
        data_packets.append((timestamp,tcp.seq,size))
print('Commands excluding get:')
for timestamp, command in commands:
    if not command.startswith('get '):
        print(timestamp,repr(command))
print('Most common commands:',Counter(c for _,c in commands).most_common(15))
bursts = []
for t, seq, size in data_packets:
    if not bursts or t-bursts[-1]['end'] > .03:
        bursts.append(dict(start=t,end=t,bytes=0,packets=0))
    bursts[-1]['end'] = t
    bursts[-1]['bytes'] += size
    bursts[-1]['packets'] += 1
for burst in bursts:
    burst['ms'] = (burst['end']-burst['start'])*1000
print('Data bursts (gap >30ms):',bursts[:15])
(root/'analysis.json').write_text(json.dumps(dict(commands=commands,bursts=bursts),indent=2))
