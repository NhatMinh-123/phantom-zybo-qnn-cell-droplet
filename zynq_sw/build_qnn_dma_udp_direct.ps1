param(
    [string]$Workspace = 'E:\fpga\finn_build\zybo_z7_10_15micro_qnn96\vitis_udp_ws3',
    [string]$Vitis = 'E:\vivado\Vitis\2022.2'
)
$ErrorActionPreference = 'Stop'
$bsp = Join-Path $Workspace 'zybo_qnn_udp_platform\ps7_cortexa9_0\standalone_domain\bsp\ps7_cortexa9_0'
$sample = Join-Path $Vitis 'data\embeddedsw\lib\sw_apps\lwip_udp_perf_server\src'
$gcc = Join-Path $Vitis 'gnu\aarch32\nt\gcc-arm-none-eabi\bin\arm-none-eabi-gcc.exe'
$ld = Join-Path $Workspace 'qnn_dma_udp\src\lscript.ld'
$output = Join-Path $Workspace 'qnn_dma_udp\Debug'
New-Item -ItemType Directory -Force -Path $output | Out-Null
$elf = Join-Path $output 'qnn_dma_udp.elf'
& $gcc '-O2' '-g' '-mcpu=cortex-a9' '-mfpu=vfpv3' '-mfloat-abi=hard' `
    "-I$bsp\include" "-I$PSScriptRoot" "-I$sample" `
    "$PSScriptRoot\qnn_dma_udp.c" "$sample\platform_zynq.c" `
    "$PSScriptRoot\rtl8211f_status_compat.c" '-Wl,--wrap=XEmacPs_PhyRead' `
    "-specs=$Workspace\qnn_dma_udp\src\Xilinx.spec" `
    "-L$bsp\lib" "-T$ld" '-Wl,--defsym=_HEAP_SIZE=0x100000' `
    '-Wl,--defsym=_STACK_SIZE=0x10000' '-Wl,--start-group' `
    '-llwip4' '-lxil' '-lgcc' '-lc' '-Wl,--end-group' '-o' $elf
if ($LASTEXITCODE -ne 0) { throw 'QNN UDP compilation failed' }
Get-Item -LiteralPath $elf | Select-Object FullName,Length,LastWriteTime
