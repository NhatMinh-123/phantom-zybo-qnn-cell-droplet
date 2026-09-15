param(
    [string]$Xsa = 'E:\fpga\finn_build\zybo_z7_10_15micro_qnn96\zybo_ps_qnn_dma\zybo_ps_qnn_dma.xsa',
    [string]$Workspace = 'E:\fpga\finn_build\zybo_z7_10_15micro_qnn96\vitis_ws',
    [string]$Source = 'E:\fpga\zynq_sw\qnn_dma_uart.c',
    [string]$AppName = 'qnn_dma_uart',
    [string]$Xsct = 'E:\vivado\Vitis\2022.2\bin\xsct.bat'
)
$ErrorActionPreference = 'Stop'
$root = 'E:\fpga'
$tcl = Join-Path $root 'zynq_sw\build_qnn_dma_uart.tcl'
$env:ZYBO_QNN_XSA = $Xsa
$env:ZYBO_QNN_WS = $Workspace
$env:ZYBO_QNN_APP_SOURCE = $Source
$env:ZYBO_QNN_APP_NAME = $AppName
& $Xsct $tcl
if ($LASTEXITCODE -ne 0) { throw "Vitis build failed with exit code $LASTEXITCODE" }
