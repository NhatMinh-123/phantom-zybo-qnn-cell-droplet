set root [file normalize "E:/fpga"]
set hw [file normalize [expr {[info exists ::env(ZYBO_QNN_XSA)] ? $::env(ZYBO_QNN_XSA) : "$root/finn_build/zybo_z7_10_15micro_qnn96/zybo_ps_qnn_dma/zybo_ps_qnn_dma.xsa"}]]
set ws [file normalize [expr {[info exists ::env(ZYBO_QNN_WS)] ? $::env(ZYBO_QNN_WS) : "$root/finn_build/zybo_z7_10_15micro_qnn96/vitis_udp_ws"}]]
set app_source [file normalize "$root/zynq_sw/qnn_dma_udp.c"]

setws $ws
platform create -name zybo_qnn_udp_platform -hw $hw -proc ps7_cortexa9_0 -os standalone
domain active standalone_domain
bsp setlib -name lwip211
bsp write
platform write
platform generate
app create -name qnn_dma_udp -platform zybo_qnn_udp_platform -domain standalone_domain -template {Empty Application(C)}
file copy -force $app_source "$ws/qnn_dma_udp/src/[file tail $app_source]"
app build -name qnn_dma_udp
puts "QNN_DMA_UDP_ELF=$ws/qnn_dma_udp/Debug/qnn_dma_udp.elf"
