"""Cached-buffer Phantom SDK reader, using the SDK's documented image header."""
from pathlib import Path
import ctypes as ct
import numpy as np
import pyphantom


class ImageHeader(ct.Structure):
    _fields_ = [
        ('size', ct.c_uint32), ('width', ct.c_int32), ('height', ct.c_int32),
        ('planes', ct.c_uint16), ('bits', ct.c_uint16),
        ('compression', ct.c_uint32), ('image_bytes', ct.c_uint32),
        ('xppm', ct.c_int32), ('yppm', ct.c_int32),
        ('colors', ct.c_uint32), ('important', ct.c_uint32),
        ('black', ct.c_int32), ('white', ct.c_int32),
    ]


class Rect(ct.Structure):
    _fields_ = [
        ('left', ct.c_long), ('top', ct.c_long),
        ('right', ct.c_long), ('bottom', ct.c_long),
    ]


class NativePhantomReader:
    def __init__(
        self,
        camera,
        fast_demosaic=False,
        demosaic_algorithm=None,
        crop_rect=None,
    ):
        self.dll = ct.WinDLL(str(Path(pyphantom.__file__).parent / 'data/PhFile.Dll'))
        self.handle = ct.c_void_p(camera._live_cine._cine_handle)
        self._restore = []
        self.effective_crop_rect = None
        self.dll.PhGetCineInfo.argtypes = [ct.c_void_p, ct.c_uint32, ct.c_void_p]
        self.dll.PhGetCineInfo.restype = ct.c_int32
        self.dll.PhSetCineInfo.argtypes = [ct.c_void_p, ct.c_uint32, ct.c_void_p]
        self.dll.PhSetCineInfo.restype = ct.c_int32
        self.dll.PhGetCineImage.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.POINTER(ImageHeader)]
        self.dll.PhGetCineImage.restype = ct.c_int32
        self.processing = ct.WinDLL(str(Path(pyphantom.__file__).parent / 'data/PhInt.Dll'))
        self.processing.PhProcessImage.argtypes = [ct.c_void_p, ct.c_void_p, ct.POINTER(ImageHeader), ct.c_uint32, ct.c_void_p]
        self.processing.PhProcessImage.restype = ct.c_int32
        try:
            if fast_demosaic and demosaic_algorithm is not None:
                raise ValueError('Choose fast_demosaic or demosaic_algorithm, not both')
            if fast_demosaic:
                demosaic_algorithm = 1
            if demosaic_algorithm is not None:
                self._set_uint(211, int(demosaic_algorithm))
            if crop_rect is not None:
                left, top, right, bottom = (int(value) for value in crop_rect)
                if left < 0 or top < 0 or right <= left or bottom <= top:
                    raise ValueError(f'Invalid SDK crop rectangle: {crop_rect}')
                self._set_rect(219, Rect(left, top, right, bottom))
                self._set_uint(218, 1)
                effective = Rect()
                self.check(self.dll.PhGetCineInfo(self.handle, 219, ct.byref(effective)))
                self.effective_crop_rect = (
                    effective.left, effective.top, effective.right, effective.bottom
                )

            self.gain = ct.c_float()
            self.check(self.dll.PhGetCineInfo(self.handle, 221, ct.byref(self.gain)))
            size = ct.c_uint32()
            self.check(self.dll.PhGetCineInfo(self.handle, 400, ct.byref(size)))
            if not 0 < size.value <= 512 * 1024 * 1024:
                raise ValueError(f'Unexpected SDK buffer size: {size.value}')
            self.buffer = (ct.c_uint8 * size.value)()
            self.reduced = (ct.c_uint8 * size.value)()
            self.header = ImageHeader()
        except BaseException:
            self.close()
            raise

    def _set_uint(self, selector, value):
        previous = ct.c_uint32()
        self.check(self.dll.PhGetCineInfo(self.handle, selector, ct.byref(previous)))
        current = ct.c_uint32(value)
        self.check(self.dll.PhSetCineInfo(self.handle, selector, ct.byref(current)))
        self._restore.append((selector, ct.c_uint32(previous.value)))

    def _set_rect(self, selector, value):
        previous = Rect()
        self.check(self.dll.PhGetCineInfo(self.handle, selector, ct.byref(previous)))
        self.check(self.dll.PhSetCineInfo(self.handle, selector, ct.byref(value)))
        self._restore.append((selector, Rect(previous.left, previous.top, previous.right, previous.bottom)))

    def close(self):
        first_error = None
        while self._restore:
            selector, value = self._restore.pop()
            try:
                self.check(self.dll.PhSetCineInfo(self.handle, selector, ct.byref(value)))
            except BaseException as exc:
                first_error = first_error or exc
        if first_error is not None:
            raise first_error

    @staticmethod
    def check(status):
        if status < 0:
            raise RuntimeError(f'Phantom SDK error {status}')

    def read(self):
        self.check(self.dll.PhGetCineImage(self.handle, None, self.buffer, len(self.buffer), ct.byref(self.header)))
        h = self.header
        source = self.buffer
        if h.bits in (16, 48):
            self.check(self.processing.PhProcessImage(self.buffer, self.reduced, ct.byref(h), 1, ct.byref(self.gain)))
            source = self.reduced
        if h.bits not in (8, 24) or h.compression != 0:
            raise ValueError(f'Unsupported native image format: {h.bits} bpp, compression {h.compression}')
        width, height, channels = abs(h.width), abs(h.height), h.bits // 8
        stride = ((width * h.bits + 31) // 32) * 4
        if not width or not height or stride * height > len(self.buffer):
            raise ValueError('Invalid SDK image dimensions')
        rows = np.ctypeslib.as_array(source)[:stride * height].reshape(height, stride)
        image = rows[:, :width * channels].reshape(height, width, channels)
        # Phantom buffers are top-down even when the legacy header height is positive.
        # Own the frame before the SDK reuses its buffer on the next call.
        return image.copy()
