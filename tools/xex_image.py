#!/usr/bin/env python3
"""Writes the loaded image of a retail XEX2 (AES-encrypted, basic or no compression) for the analysis tools, without
running the game (usf4-nx made out/default_image.bin with a run of the game and USF4_DUMP_IMAGE).

  python3 tools/xex_image.py assets/game_root/default.xex out/default_image.bin

Needs the `cryptography` package. LZX-compressed (normal compression) executables are not handled.
"""
import struct
import sys

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

RETAIL_KEY = bytes.fromhex('20B185A59D28FDC340583FBB0896BF91')
DEVKIT_KEY = bytes(16)


def aes_ecb_decrypt(key, data):
    d = Cipher(algorithms.AES(key), modes.ECB()).decryptor()
    return d.update(data) + d.finalize()


def aes_cbc_decrypt(key, data):
    d = Cipher(algorithms.AES(key), modes.CBC(bytes(16))).decryptor()
    return d.update(data) + d.finalize()


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    xex = open(sys.argv[1], 'rb').read()
    if xex[:4] != b'XEX2':
        sys.exit('not an XEX2 file')
    _flags, pe_off, _res, sec_off, opt_count = struct.unpack_from('>5I', xex, 4)
    opts = {}
    for i in range(opt_count):
        key, value = struct.unpack_from('>II', xex, 24 + i * 8)
        opts[key] = value
    image_size = struct.unpack_from('>I', xex, sec_off + 4)[0]
    load_address = struct.unpack_from('>I', xex, sec_off + 0x110)[0]
    file_key = xex[sec_off + 0x150:sec_off + 0x160]

    fmt_off = opts.get(0x3FF)
    if fmt_off is None:
        sys.exit('no file format header (0x3FF)')
    info_size, encryption, compression = struct.unpack_from('>IHH', xex, fmt_off)
    payload = xex[pe_off:]
    if encryption == 1:
        payload = payload[:len(payload) & ~15]
        session = aes_ecb_decrypt(RETAIL_KEY, file_key)
        plain = aes_cbc_decrypt(session, payload)
        if compression == 1 and plain[:2] != b'MZ':
            session = aes_ecb_decrypt(DEVKIT_KEY, file_key)
            plain = aes_cbc_decrypt(session, payload)
        payload = plain

    if compression == 1:
        out = bytearray()
        pos = 0
        for i in range((info_size - 8) // 8):
            data_size, zero_size = struct.unpack_from('>II', xex, fmt_off + 8 + i * 8)
            out += payload[pos:pos + data_size]
            out += bytes(zero_size)
            pos += data_size
    elif compression == 0:
        out = bytearray(payload)
    else:
        sys.exit(f'compression type {compression} not handled (LZX)')
    if out[:2] != b'MZ':
        sys.exit('decryption failed: no MZ header')
    out = bytes(out[:image_size]).ljust(image_size, b'\0')
    open(sys.argv[2], 'wb').write(out)
    print(f'{sys.argv[2]}: load address 0x{load_address:08X}, {len(out):,} bytes')


if __name__ == '__main__':
    main()
