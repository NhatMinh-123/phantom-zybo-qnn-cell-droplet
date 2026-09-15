param(
    [string]$Bit = 'E:\fpga\finn_build\zybo_z7_10_15micro_qnn96\zybo_ps_qnn_dma\zybo_ps_qnn_dma.runs\impl_1\zybo_qnn_wrapper.bit',
    [string]$Elf = 'E:\fpga\finn_build\zybo_z7_10_15micro_qnn96\vitis_ws\qnn_dma_uart\Debug\qnn_dma_uart.elf',
    [string]$PsInit = 'E:\fpga\finn_build\zybo_z7_10_15micro_qnn96\vitis_ws\qnn_dma_uart\_ide\psinit\ps7_init.tcl',
    [string]$Xsct = 'E:\vivado\Vitis\2022.2\bin\xsct.bat'
)
$ErrorActionPreference = 'Stop'
$env:ZYBO_QNN_BIT = $Bit
$env:ZYBO_QNN_ELF = $Elf
$env:ZYBO_QNN_PSINIT = $PsInit
& $Xsct 'E:\fpga\zynq_sw\program_zybo_qnn.tcl'
if ($LASTEXITCODE -ne 0) { throw "Zybo QNN program failed with exit code $LASTEXITCODE" }
