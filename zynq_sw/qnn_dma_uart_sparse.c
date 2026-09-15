/*
 * Zybo Z7-10 sparse QNN transport for continuous dual-ROI video.
 *
 * Request:  A5 5A + 9,216 UINT8 input bytes.
 * Response: 5A A6 + record_count(u16 LE) + qnn_us(u32 LE) + records.
 * Record:   grid_index(u16 LE), slot(u8), five INT16 raw-head values (LE).
 *
 * The FINN QNN executes in programmable logic. The ARM only moves tensors,
 * filters objectness slots and serializes the sparse response.
 */
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
#define GRID_CELLS      576U
#define SLOT_COUNT      3U
#define VALUES_PER_SLOT 5U
#define VALUE_BYTES     3U
#define CELL0_MIN_RAW   22720
#define CELL1_MIN_RAW   23153
#define DROPLET_MIN_RAW 18062

static XAxiDma dma;
static XUartPs uart;
static u8 input_tensor[INPUT_BYTES] __attribute__((aligned(64)));
static u8 output_tensor[OUTPUT_BYTES] __attribute__((aligned(64)));

static void uart_read_exact(u8 *data, u32 count) {
    u32 received = 0;
    while (received < count)
        received += XUartPs_Recv(&uart, data + received, count - received);
}

static void uart_write_exact(const u8 *data, u32 count) {
    u32 sent = 0;
    while (sent < count)
        sent += XUartPs_Send(&uart, (u8 *)(data + sent), count - sent);
}

static void write_u16(u16 value) {
    u8 data[2] = {(u8)value, (u8)(value >> 8)};
    uart_write_exact(data, 2U);
}

static void write_u32(u32 value) {
    u8 data[4] = {
        (u8)value, (u8)(value >> 8), (u8)(value >> 16), (u8)(value >> 24)
    };
    uart_write_exact(data, 4U);
}

static s16 raw_value(u32 value_index) {
    u32 offset = value_index * VALUE_BYTES;
    u16 encoded = (u16)output_tensor[offset] |
                  ((u16)output_tensor[offset + 1U] << 8);
    return (s16)encoded;
}

static s16 object_threshold(u32 slot) {
    if (slot == 0U) return (s16)CELL0_MIN_RAW;
    if (slot == 1U) return (s16)CELL1_MIN_RAW;
    return (s16)DROPLET_MIN_RAW;
}

static u16 count_sparse_records(void) {
    u32 grid;
    u32 slot;
    u16 count = 0U;
    for (grid = 0U; grid < GRID_CELLS; ++grid) {
        for (slot = 0U; slot < SLOT_COUNT; ++slot) {
            u32 object_index = grid * 15U + slot * VALUES_PER_SLOT;
            if (raw_value(object_index) >= object_threshold(slot)) ++count;
        }
    }
    return count;
}

static void write_sparse_records(void) {
    u32 grid;
    u32 slot;
    u32 value;
    for (grid = 0U; grid < GRID_CELLS; ++grid) {
        for (slot = 0U; slot < SLOT_COUNT; ++slot) {
            u32 base = grid * 15U + slot * VALUES_PER_SLOT;
            if (raw_value(base) < object_threshold(slot)) continue;
            write_u16((u16)grid);
            {
                u8 slot_byte = (u8)slot;
                uart_write_exact(&slot_byte, 1U);
            }
            for (value = 0U; value < VALUES_PER_SLOT; ++value)
                write_u16((u16)raw_value(base + value));
        }
    }
}

static int dma_infer(u32 *elapsed_us) {
    int status;
    XTime started;
    XTime stopped;
    Xil_DCacheFlushRange((UINTPTR)input_tensor, INPUT_BYTES);
    Xil_DCacheInvalidateRange((UINTPTR)output_tensor, OUTPUT_BYTES);
    XTime_GetTime(&started);
    status = XAxiDma_SimpleTransfer(
        &dma, (UINTPTR)output_tensor, OUTPUT_BYTES, XAXIDMA_DEVICE_TO_DMA
    );
    if (status != XST_SUCCESS) return status;
    status = XAxiDma_SimpleTransfer(
        &dma, (UINTPTR)input_tensor, INPUT_BYTES, XAXIDMA_DMA_TO_DEVICE
    );
    if (status != XST_SUCCESS) return status;
    while (XAxiDma_Busy(&dma, XAXIDMA_DMA_TO_DEVICE) ||
           XAxiDma_Busy(&dma, XAXIDMA_DEVICE_TO_DMA)) {}
    XTime_GetTime(&stopped);
    Xil_DCacheInvalidateRange((UINTPTR)output_tensor, OUTPUT_BYTES);
    *elapsed_us = (u32)(((stopped - started) * 1000000ULL) / COUNTS_PER_SECOND);
    return XST_SUCCESS;
}

int main(void) {
    XAxiDma_Config *dma_cfg;
    XUartPs_Config *uart_cfg;
    u8 byte;
    u32 elapsed_us;
    u16 record_count;
    const u8 ok_header[2] = {0x5AU, 0xA6U};
    const u8 err_header[2] = {0x45U, 0x52U};

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

    xil_printf("Zybo QNN sparse UART ready 2000000\r\n");
    for (;;) {
        do { uart_read_exact(&byte, 1U); } while (byte != 0xA5U);
        uart_read_exact(&byte, 1U);
        if (byte != 0x5AU) continue;
        uart_read_exact(input_tensor, INPUT_BYTES);
        if (dma_infer(&elapsed_us) != XST_SUCCESS) {
            uart_write_exact(err_header, 2U);
            continue;
        }
        record_count = count_sparse_records();
        uart_write_exact(ok_header, 2U);
        write_u16(record_count);
        write_u32(elapsed_us);
        write_sparse_records();
    }
}
