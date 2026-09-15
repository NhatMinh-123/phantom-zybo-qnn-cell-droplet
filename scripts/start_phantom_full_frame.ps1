$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$python = Join-Path $root '.venv-phantom64\Scripts\python.exe'
$existing = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object {
        $_.CommandLine -like '*run_phantom_full_frame_live.py*' -or
        $_.CommandLine -like '*run_phantom_zybo_qnn_live.py*'
    }
if ($existing) {
    exit 0
}

$sessionRoot = Join-Path $root 'final_results\phantom_camera_qnn_live\full_frame_sessions'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$out = Join-Path $sessionRoot $stamp
New-Item -ItemType Directory -Force -Path $out | Out-Null
Set-Content -LiteralPath (Join-Path $sessionRoot 'latest_session.txt') -Value $out
Start-Process -FilePath $python -WorkingDirectory $root -WindowStyle Hidden `
    -ArgumentList @('-u', 'scripts/run_phantom_full_frame_live.py', '--output', $out) `
    -RedirectStandardOutput (Join-Path $out 'stdout.log') `
    -RedirectStandardError (Join-Path $out 'stderr.log') | Out-Null
