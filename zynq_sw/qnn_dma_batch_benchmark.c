/* Measure steady-state FINN throughput without UART transfer overhead. */
#include "xaxidma.h"
#include "xil_cache.h"
#include "xil_printf.h"
#include "xparameters.h"
#include "xstatus.h"
#include "xtime_l.h"
#include "xuartps.h"

#define UART_BAUD       2000000U
#define INPUT_BYTES     9216U
#define OUTPUT_BYTES    25920U
#define BATCH_FRAMES    64U
#define DMA_TIMEOUT_US  5000000U

static XAxiDma dma;
static XUartPs uart;
static u8 input_batch[BATCH_FRAMES * INPUT_BYTES] __attribute__((aligned(64)));
static u8 output_ping[OUTPUT_BYTES] __attribute__((aligned(64)));
static u8 output_pong[OUTPUT_BYTES] __attribute__((aligned(64)));

static int wait_dma(u32 direction) {
    XTime started;
    XTime now;

    XTime_GetTime(&started);
    while (XAxiDma_Busy(&dma, direction)) {
        XTime_GetTime(&now);
        if (((now - started) * 1000000ULL) / COUNTS_PER_SECOND > DMA_TIMEOUT_US)
            return XST_FAILURE;
    }
    return XST_SUCCESS;
}

static void prepare_inputs(void) {
    u32 frame;
    u32 index;
    for (frame = 0U; frame < BATCH_FRAMES; ++frame) {
        u8 *dst = input_batch + frame * INPUT_BYTES;
        for (index = 0U; index < INPUT_BYTES; ++index)
            dst[index] = (u8)((index * 13U + frame * 29U) & 0xFFU);
    }
}

static int run_batch(u32 *elapsed_us, u32 *checksum) {
    XTime started;
    XTime stopped;
    u32 frame;
    int status;
    u8 *completed = output_ping;
    u8 *next = output_pong;

    Xil_DCacheFlushRange((UINTPTR)input_batch, sizeof(input_batch));
    Xil_DCacheInvalidateRange((UINTPTR)output_ping, OUTPUT_BYTES);
    Xil_DCacheInvalidateRange((UINTPTR)output_pong, OUTPUT_BYTES);

    XTime_GetTime(&started);
    status = XAxiDma_SimpleTransfer(
        &dma, (UINTPTR)completed, OUTPUT_BYTES, XAXIDMA_DEVICE_TO_DMA
    );
    if (status != XST_SUCCESS) return status;
    status = XAxiDma_SimpleTransfer(
        &dma, (UINTPTR)input_batch, sizeof(input_batch), XAXIDMA_DMA_TO_DEVICE
    );
    if (status != XST_SUCCESS) return status;

    *checksum = 0U;
    for (frame = 0U; frame < BATCH_FRAMES; ++frame) {
        u32 index;
        if (wait_dma(XAXIDMA_DEVICE_TO_DMA) != XST_SUCCESS)
            return XST_FAILURE;
        Xil_DCacheInvalidateRange((UINTPTR)completed, OUTPUT_BYTES);

        if (frame + 1U < BATCH_FRAMES) {
            status = XAxiDma_SimpleTransfer(
                &dma, (UINTPTR)next, OUTPUT_BYTES, XAXIDMA_DEVICE_TO_DMA
            );
            if (status != XST_SUCCESS) return status;
        }
        for (index = 0U; index < 64U; ++index)
            *checksum = (*checksum * 33U) ^ completed[index];
        {
            u8 *swap = completed;
            completed = next;
            next = swap;
        }
    }
    if (wait_dma(XAXIDMA_DMA_TO_DEVICE) != XST_SUCCESS)
        return XST_FAILURE;
    XTime_GetTime(&stopped);
    *elapsed_us = (u32)(((stopped - started) * 1000000ULL) / COUNTS_PER_SECOND);
    return XST_SUCCESS;
}

int main(void) {
    XAxiDma_Config *dma_cfg;
    XUartPs_Config *uart_cfg;
    u32 elapsed_us;
    u32 checksum;
    u8 command;
    int status;

    uart_cfg = XUartPs_LookupConfig(XPAR_XUARTPS_0_DEVICE_ID);
    if (!uart_cfg ||
        XUartPs_CfgInitialize(&uart, uart_cfg, uart_cfg->BaseAddress) != XST_SUCCESS)
        return XST_FAILURE;
    if (XUartPs_SetBaudRate(&uart, UART_BAUD) != XST_SUCCESS)
        return XST_FAILURE;

    dma_cfg = XAxiDma_LookupConfig(XPAR_AXI_DMA_0_DEVICE_ID);
    if (!dma_cfg || XAxiDma_CfgInitialize(&dma, dma_cfg) != XST_SUCCESS)
        return XST_FAILURE;
    if (XAxiDma_HasSg(&dma)) return XST_FAILURE;

    prepare_inputs();
    xil_printf("QNN_BATCH_BENCHMARK_READY send=B\r\n");
    do {
        while (XUartPs_Recv(&uart, &command, 1U) == 0U) {}
    } while (command != (u8)'B');
    xil_printf("QNN_BATCH_BENCHMARK_BEGIN frames=%u\r\n", BATCH_FRAMES);
    status = run_batch(&elapsed_us, &checksum);
    if (status != XST_SUCCESS) {
        xil_printf("QNN_BATCH_BENCHMARK_FAIL status=%d\r\n", status);
        return status;
    }
    xil_printf(
        "QNN_BATCH_BENCHMARK_PASS frames=%u total_us=%u roi_fps_x100=%u "
        "dual_roi_fps_x100=%u checksum=%08x\r\n",
        BATCH_FRAMES,
        elapsed_us,
        (u32)(((u64)BATCH_FRAMES * 100000000ULL) / elapsed_us),
        (u32)(((u64)BATCH_FRAMES * 50000000ULL) / elapsed_us),
        checksum
    );
    for (;;) {}
}
