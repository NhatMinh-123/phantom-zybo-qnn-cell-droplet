/*
 * Zybo Z7-10 Ethernet transport for the 15 um FINN QNN.
 *
 * The Cortex-A9 receives one quantized 96x96 ROI as UDP chunks, runs the
 * existing AXI DMA -> FINN accelerator -> AXI DMA path, then returns sparse
 * raw-head records tagged with the originating frame and ROI identifiers.
 */
#include <string.h>

#include "xaxidma.h"
#include "xil_cache.h"
#include "xil_printf.h"
#include "xparameters.h"
#include "xstatus.h"
#include "xtime_l.h"

#include "lwip/init.h"
#include "lwip/ip_addr.h"
#include "lwip/pbuf.h"
#include "lwip/timeouts.h"
#include "lwip/udp.h"
#include "lwip/tcp.h"
#include "lwip/priv/tcp_priv.h"
#include "lwip/etharp.h"
#include "netif/xadapter.h"
#include "platform.h"

#define UDP_PORT 50123U
#define HEADER_BYTES 24U
#define MAX_PAYLOAD_BYTES 1400U
#define INPUT_BYTES 9216U
#define OUTPUT_BYTES 25920U
#define GRID_CELLS 576U
#define SLOT_COUNT 3U
#define VALUES_PER_SLOT 5U
#define VALUE_BYTES 3U
#define RECORD_BYTES 13U
#define MAX_RECORD_BYTES (GRID_CELLS * SLOT_COUNT * RECORD_BYTES)
#define MAX_INPUT_CHUNKS ((INPUT_BYTES + MAX_PAYLOAD_BYTES - 1U) / MAX_PAYLOAD_BYTES)

#define MAGIC0 ((u8)'Z')
#define MAGIC1 ((u8)'Q')
#define PROTOCOL_VERSION 1U
#define MESSAGE_ROI 1U
#define MESSAGE_RESULT 2U
#define MESSAGE_HELLO 3U
#define MESSAGE_HELLO_REPLY 4U

#define CELL0_MIN_RAW 22720
#define CELL1_MIN_RAW 23153
#define DROPLET_MIN_RAW 18062

static XAxiDma dma;
struct netif server_netif;
extern volatile int TcpFastTmrFlag;
extern volatile int TcpSlowTmrFlag;
static struct udp_pcb *server_pcb;
static u8 input_tensor[INPUT_BYTES] __attribute__((aligned(64)));
static u8 output_tensor[OUTPUT_BYTES] __attribute__((aligned(64)));
static u8 sparse_records[MAX_RECORD_BYTES] __attribute__((aligned(64)));
static u8 chunks_seen[MAX_INPUT_CHUNKS];
static u32 assembling_frame;
static u8 assembling_roi;
static u8 expected_chunks;
static u8 received_chunks;
static ip_addr_t assembling_address;
static u16 assembling_port;

static u16 read_u16(const u8 *data) {
    return (u16)data[0] | ((u16)data[1] << 8);
}

static u32 read_u32(const u8 *data) {
    return (u32)data[0] | ((u32)data[1] << 8) |
           ((u32)data[2] << 16) | ((u32)data[3] << 24);
}

static void write_u16(u8 *data, u16 value) {
    data[0] = (u8)value;
    data[1] = (u8)(value >> 8);
}

static void write_u32(u8 *data, u32 value) {
    data[0] = (u8)value;
    data[1] = (u8)(value >> 8);
    data[2] = (u8)(value >> 16);
    data[3] = (u8)(value >> 24);
}

static void reset_assembly(u32 frame_id, u8 roi_id, u8 chunk_count) {
    assembling_frame = frame_id;
    assembling_roi = roi_id;
    expected_chunks = chunk_count;
    received_chunks = 0U;
    memset(chunks_seen, 0, sizeof(chunks_seen));
}

static s16 raw_value(u32 value_index) {
    u32 offset = value_index * VALUE_BYTES;
    return (s16)((u16)output_tensor[offset] |
                 ((u16)output_tensor[offset + 1U] << 8));
}

static s16 object_threshold(u32 slot) {
    if (slot == 0U) return (s16)CELL0_MIN_RAW;
    if (slot == 1U) return (s16)CELL1_MIN_RAW;
    return (s16)DROPLET_MIN_RAW;
}

static u16 make_sparse_records(void) {
    u32 grid;
    u32 slot;
    u32 value;
    u16 count = 0U;
    u8 *destination = sparse_records;
    for (grid = 0U; grid < GRID_CELLS; ++grid) {
        for (slot = 0U; slot < SLOT_COUNT; ++slot) {
            u32 base = grid * 15U + slot * VALUES_PER_SLOT;
            if (raw_value(base) < object_threshold(slot)) continue;
            write_u16(destination, (u16)grid);
            destination[2] = (u8)slot;
            for (value = 0U; value < VALUES_PER_SLOT; ++value) {
                write_u16(destination + 3U + value * 2U,
                          (u16)raw_value(base + value));
            }
            destination += RECORD_BYTES;
            ++count;
        }
    }
    return count;
}

static int dma_infer(u32 *elapsed_us) {
    int status;
    XTime started;
    XTime stopped;
    Xil_DCacheFlushRange((UINTPTR)input_tensor, INPUT_BYTES);
    Xil_DCacheInvalidateRange((UINTPTR)output_tensor, OUTPUT_BYTES);
    XTime_GetTime(&started);
    status = XAxiDma_SimpleTransfer(&dma, (UINTPTR)output_tensor, OUTPUT_BYTES,
                                    XAXIDMA_DEVICE_TO_DMA);
    if (status != XST_SUCCESS) return status;
    status = XAxiDma_SimpleTransfer(&dma, (UINTPTR)input_tensor, INPUT_BYTES,
                                    XAXIDMA_DMA_TO_DEVICE);
    if (status != XST_SUCCESS) return status;
    while (XAxiDma_Busy(&dma, XAXIDMA_DMA_TO_DEVICE) ||
           XAxiDma_Busy(&dma, XAXIDMA_DEVICE_TO_DMA)) {
        XTime_GetTime(&stopped);
        if (stopped - started > COUNTS_PER_SECOND) {
            xil_printf("QNN DMA timeout\r\n");
            XAxiDma_Reset(&dma);
            return XST_FAILURE;
        }
    }
    XTime_GetTime(&stopped);
    Xil_DCacheInvalidateRange((UINTPTR)output_tensor, OUTPUT_BYTES);
    *elapsed_us = (u32)(((stopped - started) * 1000000ULL) /
                        COUNTS_PER_SECOND);
    return XST_SUCCESS;
}

static void send_packet(const ip_addr_t *address, u16 port, u8 message_type,
                        u32 frame_id, u8 roi_id, u8 chunk_index,
                        u8 chunk_count, u16 total_bytes, u16 offset,
                        u16 record_count, u32 qnn_us, const u8 *payload,
                        u16 payload_bytes) {
    u8 header[HEADER_BYTES];
    struct pbuf *packet;
    memset(header, 0, sizeof(header));
    header[0] = MAGIC0;
    header[1] = MAGIC1;
    header[2] = PROTOCOL_VERSION;
    header[3] = message_type;
    write_u32(header + 4U, frame_id);
    header[8] = roi_id;
    header[9] = chunk_index;
    header[10] = chunk_count;
    write_u16(header + 12U, total_bytes);
    write_u16(header + 14U, offset);
    write_u16(header + 16U, payload_bytes);
    write_u16(header + 18U, record_count);
    write_u32(header + 20U, qnn_us);
    packet = pbuf_alloc(PBUF_TRANSPORT, HEADER_BYTES + payload_bytes, PBUF_RAM);
    if (packet == NULL) return;
    (void)pbuf_take(packet, header, HEADER_BYTES);
    if (payload_bytes != 0U) {
        (void)pbuf_take_at(packet, payload, payload_bytes, HEADER_BYTES);
    }
    (void)udp_sendto(server_pcb, packet, address, port);
    pbuf_free(packet);
}

static void send_result(const ip_addr_t *address, u16 port, u32 frame_id,
                        u8 roi_id, u16 record_count, u32 qnn_us) {
    u16 total_bytes = (u16)(record_count * RECORD_BYTES);
    u8 chunk_count = (u8)((total_bytes + MAX_PAYLOAD_BYTES - 1U) /
                          MAX_PAYLOAD_BYTES);
    u8 chunk_index;
    if (chunk_count == 0U) chunk_count = 1U;
    for (chunk_index = 0U; chunk_index < chunk_count; ++chunk_index) {
        u16 offset = (u16)(chunk_index * MAX_PAYLOAD_BYTES);
        u16 payload_bytes = total_bytes > offset
                                ? (u16)(total_bytes - offset)
                                : 0U;
        if (payload_bytes > MAX_PAYLOAD_BYTES) {
            payload_bytes = MAX_PAYLOAD_BYTES;
        }
        send_packet(address, port, MESSAGE_RESULT, frame_id, roi_id,
                    chunk_index, chunk_count, total_bytes, offset,
                    record_count, qnn_us, sparse_records + offset,
                    payload_bytes);
    }
}

static void receive_packet(void *argument, struct udp_pcb *pcb,
                           struct pbuf *packet, const ip_addr_t *address,
                           u16 port) {
    u8 header[HEADER_BYTES];
    u32 frame_id;
    u8 roi_id;
    u8 chunk_index;
    u8 chunk_count;
    u16 total_bytes;
    u16 offset;
    u16 payload_bytes;
    u32 qnn_us;
    u16 record_count;
    ip_addr_t peer_address;
    (void)argument;
    (void)pcb;
    if (packet == NULL) return;
    ip_addr_copy(peer_address, *address);
    if (packet->tot_len < HEADER_BYTES ||
        pbuf_copy_partial(packet, header, HEADER_BYTES, 0U) != HEADER_BYTES ||
        header[0] != MAGIC0 || header[1] != MAGIC1 ||
        header[2] != PROTOCOL_VERSION) {
        pbuf_free(packet);
        return;
    }
    frame_id = read_u32(header + 4U);
    roi_id = header[8];
    chunk_index = header[9];
    chunk_count = header[10];
    total_bytes = read_u16(header + 12U);
    offset = read_u16(header + 14U);
    payload_bytes = read_u16(header + 16U);
    if (header[3] == MESSAGE_HELLO) {
        static const u8 reply[] = "ZYBO-QNN-UDP-READY";
        send_packet(address, port, MESSAGE_HELLO_REPLY, frame_id, 0U, 0U,
                    1U, (u16)(sizeof(reply) - 1U), 0U, 0U, 0U, reply,
                    (u16)(sizeof(reply) - 1U));
        pbuf_free(packet);
        return;
    }
    if (header[3] != MESSAGE_ROI || total_bytes != INPUT_BYTES ||
        chunk_count != MAX_INPUT_CHUNKS ||
        chunk_index >= chunk_count || payload_bytes > MAX_PAYLOAD_BYTES ||
        offset != chunk_index * MAX_PAYLOAD_BYTES ||
        payload_bytes != (chunk_index + 1U == MAX_INPUT_CHUNKS
                         ? INPUT_BYTES - offset : MAX_PAYLOAD_BYTES) ||
        (u32)offset + payload_bytes > INPUT_BYTES ||
        packet->tot_len != HEADER_BYTES + payload_bytes) {
        pbuf_free(packet);
        return;
    }
    if (frame_id != assembling_frame || roi_id != assembling_roi ||
        chunk_count != expected_chunks || port != assembling_port ||
        !ip_addr_cmp(&peer_address, &assembling_address)) {
        reset_assembly(frame_id, roi_id, chunk_count);
        ip_addr_copy(assembling_address, peer_address);
        assembling_port = port;
    }
    if (pbuf_copy_partial(packet, input_tensor + offset, payload_bytes,
                          HEADER_BYTES) != payload_bytes) {
        pbuf_free(packet);
        return;
    }
    if (chunks_seen[chunk_index] == 0U) {
        chunks_seen[chunk_index] = 1U;
        ++received_chunks;
    }
    pbuf_free(packet);
    if (received_chunks != expected_chunks) return;
    expected_chunks = 0U;
    if (dma_infer(&qnn_us) != XST_SUCCESS) return;
    record_count = make_sparse_records();
    send_result(&peer_address, port, frame_id, roi_id, record_count, qnn_us);
}

int main(void) {
    XAxiDma_Config *dma_config;
    ip_addr_t ip_address;
    ip_addr_t netmask;
    ip_addr_t gateway;
    u8 mac_address[6] = {0x00U, 0x0AU, 0x35U, 0x51U, 0x4EU, 0x10U};
    XTime last_arp, now;

    init_platform();
    dma_config = XAxiDma_LookupConfig(XPAR_AXI_DMA_0_DEVICE_ID);
    if (dma_config == NULL ||
        XAxiDma_CfgInitialize(&dma, dma_config) != XST_SUCCESS ||
        XAxiDma_HasSg(&dma)) {
        return XST_FAILURE;
    }
    IP4_ADDR(&ip_address, 100U, 100U, 100U, 2U);
    IP4_ADDR(&netmask, 255U, 255U, 0U, 0U);
    IP4_ADDR(&gateway, 0U, 0U, 0U, 0U);
    lwip_init();
    if (xemac_add(&server_netif, &ip_address, &netmask, &gateway,
                  mac_address, XPAR_XEMACPS_0_BASEADDR) == NULL) {
        return XST_FAILURE;
    }
    netif_set_default(&server_netif);
    netif_set_up(&server_netif);
    platform_enable_interrupts();
    server_pcb = udp_new();
    if (server_pcb == NULL ||
        udp_bind(server_pcb, IP_ADDR_ANY, UDP_PORT) != ERR_OK) {
        return XST_FAILURE;
    }
    udp_recv(server_pcb, receive_packet, NULL);
    XTime_GetTime(&last_arp);
    xil_printf("Zybo QNN UDP ready 100.100.100.2:%d\r\n", UDP_PORT);
    for (;;) {
        xemacif_input(&server_netif);
        if (TcpFastTmrFlag) { tcp_fasttmr(); TcpFastTmrFlag = 0; }
        if (TcpSlowTmrFlag) { tcp_slowtmr(); TcpSlowTmrFlag = 0; }
        XTime_GetTime(&now);
        if (now - last_arp >= COUNTS_PER_SECOND) {
            etharp_tmr();
            last_arp = now;
        }
    }
}
