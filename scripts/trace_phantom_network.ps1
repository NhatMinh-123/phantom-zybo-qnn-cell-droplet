$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$out = Join-Path $root 'reports\ethernet_live_20260910\stream_optimization\network_trace'
New-Item -ItemType Directory -Force -Path $out | Out-Null
Start-Transcript -Path (Join-Path $out 'trace.log') -Force
try {
    # Never stop or overwrite another active trace or broaden its filters.
    $status = & pktmon status | Out-String
    Write-Output $status
    if ($status -notmatch 'not running') { throw 'Another trace may be active; refusing to change it' }
    $filters = & pktmon filter list | Out-String
    Write-Output $filters
    if ($filters -notmatch '(No filters|Packet Filters:\s+None)') { throw 'Existing filters detected; leaving them unchanged' }
    & pktmon filter add QnnCameraStream -i 100.100.196.217 -t TCP
    try {
        & pktmon start --capture --comp nics --pkt-size 128 --file-size 32 --file-name (Join-Path $out 'camera.etl')
        if ($LASTEXITCODE -ne 0) { throw 'Cannot start packet capture' }
        try {
            & "$root\.venv-phantom64\Scripts\python.exe" "$root\scripts\benchmark_phantom_acquisition.py" --frames 20 --output $out
        } finally { & pktmon stop }
    } finally { & pktmon filter remove QnnCameraStream }
    & pktmon etl2pcap (Join-Path $out 'camera.etl') --out (Join-Path $out 'camera.pcapng')
    & pktmon etl2txt (Join-Path $out 'camera.etl') --out (Join-Path $out 'camera.txt')
} finally { Stop-Transcript }
