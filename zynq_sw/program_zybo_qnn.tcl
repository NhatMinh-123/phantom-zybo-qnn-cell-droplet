set root [file normalize "E:/fpga"]
set bit [file normalize [expr {[info exists ::env(ZYBO_QNN_BIT)] ? $::env(ZYBO_QNN_BIT) : "$root/finn_build/zybo_z7_10_15micro_qnn96/zybo_ps_qnn_dma/zybo_ps_qnn_dma.runs/impl_1/zybo_qnn_wrapper.bit"}]]
set elf [file normalize [expr {[info exists ::env(ZYBO_QNN_ELF)] ? $::env(ZYBO_QNN_ELF) : "$root/finn_build/zybo_z7_10_15micro_qnn96/vitis_ws/qnn_dma_uart/Debug/qnn_dma_uart.elf"}]]
set psinit [file normalize [expr {[info exists ::env(ZYBO_QNN_PSINIT)] ? $::env(ZYBO_QNN_PSINIT) : "$root/finn_build/zybo_z7_10_15micro_qnn96/vitis_ws/qnn_dma_uart/_ide/psinit/ps7_init.tcl"}]]

connect
targets -set -filter {name =~ "APU*"}
rst -system
after 2000
source $psinit
ps7_init
ps7_post_config
fpga -file $bit
targets -set -filter {name =~ "ARM Cortex-A9 MPCore #0"}
dow $elf
con
puts "ZYBO_QNN_PROGRAM=PASS"
