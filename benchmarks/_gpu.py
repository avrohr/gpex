"""Query CUDA-visible hardware in a short-lived process without creating contexts."""

import ctypes
import json
import subprocess


def hardware():
    driver = ctypes.CDLL("libcuda.so.1")

    def call(name, *args):
        status = getattr(driver, name)(*args)
        if status:
            raise RuntimeError(f"{name} failed with CUDA error {status}")

    call("cuInit", 0)
    count = ctypes.c_int()
    call("cuDeviceGetCount", ctypes.byref(count))
    if count.value == 0:
        raise RuntimeError("CUDA has no visible devices")
    devices = []
    for ordinal in range(count.value):
        device = ctypes.c_int()
        call("cuDeviceGet", ctypes.byref(device), ordinal)
        name = ctypes.create_string_buffer(256)
        memory = ctypes.c_size_t()
        call("cuDeviceGetName", name, len(name), device)
        call("cuDeviceTotalMem_v2", ctypes.byref(memory), device)
        properties = dict(name=name.value.decode(), memory_bytes=memory.value)
        # CUdevice_attribute constants from cuda.h; CUDA resolves visibility/order.
        for key, attribute in (
            ("multiprocessors", 16),
            ("compute_major", 75),
            ("compute_minor", 76),
        ):
            value = ctypes.c_int()
            call("cuDeviceGetAttribute", ctypes.byref(value), attribute, device)
            properties[key] = value.value
        devices.append(properties)
    release = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
    ).splitlines()[0]
    return dict(devices=devices, driver_version=release.strip())


if __name__ == "__main__":
    print(json.dumps(hardware()))
