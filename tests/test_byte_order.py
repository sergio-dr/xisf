"""Tests for the data block byte order (endianness).

Covers the byteOrder attribute of section 10.4 of the XISF 1.0 specification.
A baseline decoder shall read data blocks in both byte orders (section 7.2), and
little-endian is assumed when the attribute is absent.
"""

import sys

import numpy as np
import pytest

from xisf import XISF, XISFError

from .conftest import METADATA, image_element

# The dtype that serializes a given sample format in a given byte order
BYTE_ORDERED = {
    "big": {"Float32": ">f4", "Float64": ">f8", "UInt16": ">u2", "UInt32": ">u4"},
    "little": {"Float32": "<f4", "Float64": "<f8", "UInt16": "<u2", "UInt32": "<u4"},
}

# numpy spells the machine's byte order as "<" or ">", while sys reports
# "little" or "big"
MACHINE = "<" if sys.byteorder == "little" else ">"
FOREIGN = ">" if MACHINE == "<" else "<"


@pytest.fixture
def write_image(tmp_path):
    def _write(im, byte_order, pixel_storage, name="bo"):
        xml = (
            f"<xisf xmlns='http://www.pixinsight.com/xisf' version='1.0'>"
            f"{METADATA}"
            f"{image_element(im, pixel_storage=pixel_storage, byte_order=byte_order)}"
            f"</xisf>"
        ).encode()
        head = (
            XISF._signature
            + len(xml).to_bytes(4, "little")
            + (0).to_bytes(4, "little")
            + xml
        )
        head += b"\0" * ((4096 - len(head) % 4096) % 4096)
        path = tmp_path / f"{name}.xisf"
        path.write_bytes(head)
        return path

    return _write


@pytest.mark.parametrize("storage", ["Planar", "Normal"])
@pytest.mark.parametrize("byte_order", ["big", "little"])
def test_reader_honors_byte_order(write_image, byte_order, storage):
    """A baseline decoder shall read data blocks in both byte orders (spec 7.2).

    The serialized bytes are in byte_order, and the array returned to the caller
    is in the machine's byte order with the correct values.
    """
    native = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)
    # Serialize the values as the requested byte order
    im = native.astype(BYTE_ORDERED[byte_order]["Float32"])
    # numpy reports a dtype with the machine's marker as "=", so compare str
    assert im.dtype.str == BYTE_ORDERED[byte_order]["Float32"]

    path = write_image(im, byte_order, storage)
    back = XISF(str(path)).read_image(0, data_format="channels_first")

    assert back.dtype.isnative, "array shall be in the machine's byte order"
    assert back.dtype == np.float32
    np.testing.assert_array_equal(back, native.transpose(2, 0, 1))


def test_absent_byte_order_is_little_endian(write_image):
    """Little-endian is assumed when the attribute is absent (spec 10.4)."""
    im = np.arange(2 * 3, dtype=np.float32).reshape(2, 3, 1)
    path = write_image(im.astype("<f4"), None, "Planar")
    np.testing.assert_array_equal(XISF(str(path)).read_image(0), im)


def test_explicit_little_is_read_as_little(write_image):
    """The little value is redundant but valid, and shall be honored."""
    native = np.arange(2 * 3, dtype=np.float32).reshape(2, 3, 1)
    path = write_image(native.astype("<f4"), "little", "Planar")
    np.testing.assert_array_equal(XISF(str(path)).read_image(0), native)


def test_both_byte_orders_yield_the_same_values(write_image):
    """The two byte orders are distinct encodings of the same image."""
    native = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)
    big = XISF(
        str(write_image(native.astype(">f4"), "big", "Planar", "be"))
    ).read_image(0, data_format="channels_first")
    little = XISF(
        str(write_image(native.astype("<f4"), "little", "Planar", "le"))
    ).read_image(0, data_format="channels_first")
    np.testing.assert_array_equal(big, little)
    np.testing.assert_array_equal(big, native.transpose(2, 0, 1))


def test_byte_order_is_recorded_in_metadata(write_image):
    """The byteOrder attribute is exposed, defaulting to little-endian."""
    im = np.arange(2 * 3, dtype=np.float32).reshape(2, 3, 1)
    for order, expected in (("big", "big"), ("little", "little"), (None, "little")):
        path = write_image(im.astype("<f4"), order, "Planar", f"m{order}")
        assert XISF(str(path)).get_images_metadata()[0]["byteOrder"] == expected


def test_invalid_byte_order_is_reported():
    """Only 'big' and 'little' are valid endianness values (spec 10.4)."""
    with pytest.raises(XISFError, match="byteOrder"):
        XISF._dtype_for_byte_order(np.dtype("float32"), "middle")


def test_single_byte_sample_format_ignores_byte_order():
    """Endianness is immaterial for single-byte blocks (spec 10.4)."""
    # UInt8 is returned unchanged in both byte orders
    for order in ("big", "little"):
        assert XISF._dtype_for_byte_order(np.dtype("uint8"), order) == np.dtype("uint8")


def test_swapped_block_is_writable(write_image):
    """A byte-swapped block is a copy, so it is writable."""
    im = np.arange(2 * 3, dtype=np.float32).reshape(2, 3, 1)
    path = write_image(im.astype(">f4"), "big", "Planar")
    back = XISF(str(path)).read_image(0)
    assert back.flags.writeable, "a byte-swapped block is a copy, not a view"


def test_native_block_remains_a_read_only_view(tmp_path):
    """A native-endian block is still a zero-copy read-only view."""
    im = np.arange(2 * 3, dtype=np.float32).reshape(2, 3)
    path = tmp_path / "native.xisf"
    XISF.write(str(path), im, "t")
    back = XISF(str(path)).read_image(0, data_format="channels_first")
    assert not back.flags.writeable
    assert not back.flags.owndata
    np.testing.assert_array_equal(back[0], im)


@pytest.mark.parametrize("sample_format", ["Float32", "Float64", "UInt16", "UInt32"])
def test_writer_rejects_foreign_byte_order(tmp_path, sample_format):
    """The data block is written as-is, so a foreign byte order is rejected."""
    im = np.arange(2 * 3, dtype=BYTE_ORDERED["big" if FOREIGN == ">" else "little"][
        sample_format
    ]).reshape(2, 3)
    assert im.dtype.str[0] == FOREIGN
    with pytest.raises(XISFError, match="byte order"):
        XISF.write(str(tmp_path / "foreign.xisf"), im, "t")


def test_writer_accepts_the_documented_conversion(tmp_path):
    """The conversion named in the error message is enough to write the file."""
    im = np.arange(2 * 3, dtype=FOREIGN + "f4").reshape(2, 3)
    converted = im.astype(im.dtype.newbyteorder("="))
    assert converted.dtype.str == MACHINE + "f4"

    path = tmp_path / "converted.xisf"
    XISF.write(str(path), converted, "t")
    back = XISF(str(path)).read_image(0, data_format="channels_first")
    np.testing.assert_array_equal(back[0], converted)
    np.testing.assert_array_equal(back[0], im)


def test_writer_accepts_explicit_native_marker(tmp_path):
    """An explicit marker matching the machine is the machine's byte order."""
    im = np.arange(2 * 3, dtype=MACHINE + "f4").reshape(2, 3)
    path = tmp_path / "explicit.xisf"
    XISF.write(str(path), im, "t")
    back = XISF(str(path)).read_image(0, data_format="channels_first")
    np.testing.assert_array_equal(back[0], im.astype(np.float32))


def test_uint8_is_never_rejected(tmp_path):
    """A single-byte sample format has no byte order to be foreign to."""
    im = np.arange(2 * 3, dtype=np.uint8).reshape(2, 3)
    path = tmp_path / "u8.xisf"
    XISF.write(str(path), im, "t")
    np.testing.assert_array_equal(
        XISF(str(path)).read_image(0, data_format="channels_first")[0], im
    )
    # even when explicitly marked as the other byte order
    im_be = np.arange(2 * 3, dtype=">u1").reshape(2, 3)
    XISF.write(str(tmp_path / "u8be.xisf"), im_be, "t")
    np.testing.assert_array_equal(
        XISF(str(tmp_path / "u8be.xisf")).read_image(0, data_format="channels_first")[0],
        im_be,
    )
