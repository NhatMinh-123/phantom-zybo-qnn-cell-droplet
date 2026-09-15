import time
import unittest
from scripts.zybo_qnn_udp_protocol import ZyboQnnUdpClient, make_packet, unpack_packet


class FakeSocket:
    def __init__(self, packets):
        self.packets = iter(packets)

    def settimeout(self, timeout):
        pass

    def recvfrom(self, size):
        return next(self.packets), ('100.100.100.2', 50123)


def fragment(index, offset, payload, total=1417, count=2, records=109, micros=8500):
    return make_packet(2, 17, 1, index, count, total, offset, records, micros, payload)


class ProtocolTests(unittest.TestCase):
    def receive(self, packets):
        client = ZyboQnnUdpClient.__new__(ZyboQnnUdpClient)
        client.timeout = 1
        client.socket = FakeSocket(packets)
        return client._receive_result(17, 1, time.perf_counter())

    def test_out_of_order_duplicate(self):
        first, last = b'a' * 1400, b'b' * 17
        result = self.receive([fragment(1,1400,last), fragment(1,1400,last), fragment(0,0,first)])
        self.assertEqual(result.records, first + last)

    def test_empty_result(self):
        result = self.receive([fragment(0,0,b'',0,1,0)])
        self.assertEqual(result.record_count, 0)

    def test_gap_rejected(self):
        with self.assertRaises(RuntimeError):
            self.receive([fragment(0,1,b'a'*1400)])

    def test_mixed_inferences_rejected(self):
        with self.assertRaises(RuntimeError):
            self.receive([fragment(0,0,b'a'*1400), fragment(1,1400,b'b'*17,micros=9999)])

    def test_conflicting_duplicate_rejected(self):
        with self.assertRaises(RuntimeError):
            self.receive([fragment(1,1400,b'a'*17), fragment(1,1400,b'b'*17)])

    def test_truncated_packet(self):
        with self.assertRaises(ValueError):
            unpack_packet(fragment(0,0,b'a'*1400)[:-1])


if __name__ == '__main__':
    unittest.main()
