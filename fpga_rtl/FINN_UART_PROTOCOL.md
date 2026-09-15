# FINN UART functional-test protocol

Protocol nay chi dung de xac minh mot frame tren Arty S7-25. UART 115200
khong du bang thong cho camera realtime.

## PC gui den FPGA

| Offset | Kich thuoc | Noi dung |
|---:|---:|---|
| 0 | 4 | ASCII `CDQ1` |
| 4 | 2 | Frame ID, unsigned little-endian |
| 6 | 27.648 | UINT8 input theo thu tu NHWC |
| 27.654 | 2 | Tong tat ca byte payload modulo 65536, little-endian |

Pixel phai duoc luong tu bang `qnn/fpga_io.py`; khong gui truc tiep byte anh
goc.

Wrapper ghi du 27.648 byte vao BRAM va chi nha reset FINN sau khi checksum
dung. Vi vay so chu ky trong response khong bao gom thoi gian UART nhan anh.

## FPGA tra ve PC

| Offset | Kich thuoc | Noi dung |
|---:|---:|---|
| 0 | 4 | ASCII `RDQ1` |
| 4 | 2 | Frame ID, unsigned little-endian |
| 6 | 1 | Status |
| 7 | 4 | Payload length, unsigned little-endian |
| 11 | 4 | So chu ky tu AXIS input dau tien den AXIS output cuoi, unsigned little-endian |
| 15 | 51.840 | INT16 accumulator NHWC, low byte truoc |
| 51.855 | 2 | Tong tat ca byte payload modulo 65536 |

Status:

- `0`: thanh cong.
- `1`: checksum input sai.
- `2`: du phong cho loi input buffer.
- `3`: so byte AXI input duoc chap nhan khong du.

Neu status khac 0, payload length bang 0 va hai byte checksum bang 0.

## Thu nghiem

Tao packet ma khong can board:

```powershell
.\.venv-yolo\Scripts\python.exe scripts\send_frame_finn_uart.py --dry-run
```

Bitstream da place/route:

```text
fpga_build/cell_droplet_finn_uart/cell_droplet_finn_uart_top.bit
```

Sau khi cam Arty S7-25, dong Hardware Manager dang giu target neu co. Lenh sau
tu nap bitstream qua JTAG, tim UART va kiem tra mot frame:

```powershell
powershell -ExecutionPolicy Bypass -File fpga_build\run_finn_uart_hardware.ps1 `
  -Port COM12
```

Co the bo tham so `-Port` de tu tim cong Digilent USB-UART, hoac dung
`-SkipProgram` neu bitstream da duoc nap.

Mac dinh script tinh accumulator golden tu checkpoint W4A4 va yeu cau tensor
FPGA trung tung gia tri. Ket qua dung phai co dong:

```text
Hardware vs checkpoint PASS: mismatches=0, max_abs_error=0
```

Dung `--no-golden-check` chi khi can doc output cua mot bitstream khac
checkpoint trong manifest.

LED0 bao clock/reset da san sang, LED1 bao dang nhan/xu ly frame, LED2 bao
dang gui response va LED3 bao loi da duoc giu lai.
