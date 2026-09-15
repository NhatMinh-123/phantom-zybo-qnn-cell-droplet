$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$python = Join-Path $root '.venv-phantom64\Scripts\python.exe'
$existing = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*run_phantom_zybo_qnn_live.py*' }
if ($existing) {
    exit 0
}

$resultRoot = Join-Path $root 'final_results\phantom_camera_qnn_live\live_sessions'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$out = Join-Path $resultRoot $stamp
New-Item -ItemType Directory -Force -Path $out | Out-Null
Set-Content -LiteralPath (Join-Path $resultRoot 'latest_session.txt') -Value $out
$process = Start-Process -FilePath $python -WorkingDirectory $root -WindowStyle Hidden `
    -ArgumentList @('-u', 'scripts/run_phantom_zybo_qnn_live.py', '--duration-sec', '0', '--reader', 'native-crop-fast', '--output', $out) `
    -RedirectStandardOutput (Join-Path $out 'stdout.log') `
    -RedirectStandardError (Join-Path $out 'stderr.log') -PassThru
$process | Select-Object Id,ProcessName
