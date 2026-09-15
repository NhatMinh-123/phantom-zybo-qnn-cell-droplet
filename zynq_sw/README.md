# Zybo Z7-10 QNN hardware test

`qnn_dma_uart.c` is a bare-metal PS application for the PS-QNN-DMA hardware
platform. It receives one 96x96 QNN tensor over the Zybo USB UART (COM13),
starts the AXI DMA, and returns the 15x24x24 FINN raw head.

The packet protocol is deliberately binary and fixed length:

- input: `A5 5A` followed by 9,216 quantized input bytes;
- output success: `5A A5` followed by 25,920 output bytes;
- output error: `45 52`.

At 115200 baud this validates actual PL inference but cannot be video
realtime. Realtime camera operation needs the camera frame path in the Zynq
(Ethernet/USB/HDMI or PL capture) rather than serializing every frame.
