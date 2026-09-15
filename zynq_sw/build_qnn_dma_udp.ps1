param(
    [string]$Xsa = 'E:\fpga\finn_build\zybo_z7_10_15micro_qnn96\zybo_ps_qnn_dma\zybo_ps_qnn_dma.xsa',
    [string]$Workspace = 'E:\fpga\finn_build\zybo_z7_10_15micro_qnn96\vitis_udp_ws',
    [string]$Xsct = 'E:\vivado\Vitis\2022.2\bin\xsct.bat'
)
$ErrorActionPreference = 'Stop'
$env:ZYBO_QNN_XSA = $Xsa
$env:ZYBO_QNN_WS = $Workspace
& $Xsct 'E:\fpga\zynq_sw\build_qnn_dma_udp.tcl'
if ($LASTEXITCODE -ne 0) {
    throw "Zybo QNN UDP firmware build failed with exit code $LASTEXITCODE"
}
