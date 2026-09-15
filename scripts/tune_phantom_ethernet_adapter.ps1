param(
    [ValidateSet('apply', 'restore')]
    [string]$Mode = 'apply'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$reportDir = Join-Path $root 'reports\ethernet_live_20260910\stream_optimization\adapter'
$beforePath = Join-Path $reportDir 'before.json'
$afterPath = Join-Path $reportDir 'after.json'
New-Item -ItemType Directory -Force -Path $reportDir | Out-Null

$adapter = Get-NetAdapter -Name 'Ethernet'
if ($adapter.InterfaceDescription -notlike 'Realtek PCIe GbE Family Controller*') {
    throw "Refusing to tune unexpected adapter: $($adapter.InterfaceDescription)"
}

$keywords = @('*EEE', 'EnableGreenEthernet', 'PowerSavingMode', 'GigaLite')

function Get-TuningState {
    $properties = foreach ($keyword in $keywords) {
        Get-NetAdapterAdvancedProperty -Name $adapter.Name -RegistryKeyword $keyword |
            Select-Object DisplayName, DisplayValue, RegistryKeyword, RegistryValue
    }
    [pscustomobject]@{
        captured_at = (Get-Date).ToString('o')
        adapter = $adapter.Name
        description = $adapter.InterfaceDescription
        link_speed = (Get-NetAdapter -Name $adapter.Name).LinkSpeed
        properties = @($properties)
    }
}

if ($Mode -eq 'apply') {
    if (-not (Test-Path -LiteralPath $beforePath)) {
        Get-TuningState | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $beforePath
    }
    foreach ($keyword in $keywords) {
        Set-NetAdapterAdvancedProperty -Name $adapter.Name -RegistryKeyword $keyword `
            -RegistryValue 0 -NoRestart
    }
} else {
    if (-not (Test-Path -LiteralPath $beforePath)) {
        throw "Saved adapter state not found: $beforePath"
    }
    $saved = Get-Content -LiteralPath $beforePath -Raw | ConvertFrom-Json
    foreach ($property in $saved.properties) {
        Set-NetAdapterAdvancedProperty -Name $adapter.Name `
            -RegistryKeyword $property.RegistryKeyword `
            -RegistryValue ([int]$property.RegistryValue[0]) -NoRestart
    }
}

Restart-NetAdapter -Name $adapter.Name -Confirm:$false
$deadline = (Get-Date).AddSeconds(20)
do {
    Start-Sleep -Milliseconds 500
    $adapter = Get-NetAdapter -Name $adapter.Name
} while ($adapter.Status -ne 'Up' -and (Get-Date) -lt $deadline)
if ($adapter.Status -ne 'Up') {
    throw 'Ethernet adapter did not return to Up state'
}
Get-TuningState | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $afterPath
Get-Content -LiteralPath $afterPath
