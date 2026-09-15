set root [file normalize "E:/fpga"]
set hw [file normalize [expr {[info exists ::env(ZYBO_QNN_XSA)] ? $::env(ZYBO_QNN_XSA) : "$root/finn_build/zybo_z7_10_15micro_qnn96/zybo_ps_qnn_dma/zybo_ps_qnn_dma.xsa"}]]
set ws [file normalize [expr {[info exists ::env(ZYBO_QNN_WS)] ? $::env(ZYBO_QNN_WS) : "$root/finn_build/zybo_z7_10_15micro_qnn96/vitis_ws"}]]
set app_source [file normalize [expr {[info exists ::env(ZYBO_QNN_APP_SOURCE)] ? $::env(ZYBO_QNN_APP_SOURCE) : "$root/zynq_sw/qnn_dma_uart.c"}]]
set app_name [expr {[info exists ::env(ZYBO_QNN_APP_NAME)] ? $::env(ZYBO_QNN_APP_NAME) : "qnn_dma_uart"}]

setws $ws
platform create -name zybo_qnn_platform -hw $hw -proc ps7_cortexa9_0 -os standalone
platform write
platform generate

app create -name $app_name -platform zybo_qnn_platform -domain standalone_domain -template {Empty Application(C)}
file copy -force $app_source "$ws/$app_name/src/[file tail $app_source]"
app build -name $app_name
puts "QNN_DMA_UART_ELF=$ws/$app_name/Debug/$app_name.elf"
