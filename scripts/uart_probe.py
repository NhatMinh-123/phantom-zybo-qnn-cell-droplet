import argparse
import sys
import time

try:
    import serial
except ImportError:
    serial = None


def parse_args():
    parser = argparse.ArgumentParser(description="Read and echo-test the Arty S7 UART probe.")
    parser.add_argument("--port", default="COM12")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--send", default="abc123\n", help="Text to send after opening the port.")
    parser.add_argument("--no-send", action="store_true")
    return parser.parse_args()


def printable(byte_value):
    if 32 <= byte_value <= 126:
        return chr(byte_value)
    if byte_value == 10:
        return "\\n"
    if byte_value == 13:
        return "\\r"
    return "."


def main():
    if serial is None:
        print("pyserial is not installed. Run: python -m pip install pyserial", file=sys.stderr)
        sys.exit(2)

    args = parse_args()
    received = []

    with serial.Serial(args.port, args.baudrate, timeout=0.1, write_timeout=1.0) as ser:
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        time.sleep(0.2)

        if not args.no_send and args.send:
            ser.write(args.send.encode("ascii"))
            ser.flush()
            print(f"sent: {args.send!r}")

        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            data = ser.read(1)
            if data:
                value = data[0]
                received.append(value)
                print(f"rx: 0x{value:02X} '{printable(value)}'")

    if not received:
        print("No UART bytes received.")
        sys.exit(1)

    text = "".join(printable(value) for value in received)
    print(f"received {len(received)} byte(s): {text}")


if __name__ == "__main__":
    main()
