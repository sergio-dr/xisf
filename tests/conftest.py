"""Shared helpers for building hand-crafted XISF headers in tests."""

import base64
import zlib

import lz4.block
import numpy as np
import pytest
import zstandard

from xisf import XISF

XISF_NS = "http://www.pixinsight.com/xisf"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"

# Root element attributes as emitted by XISF.write(), with both namespace
# declarations so hand-built headers can also use the xisf: prefix.
ROOT_ATTRS = (
    f'xmlns="{XISF_NS}" '
    f'xmlns:xsi="{XSI_NS}" '
    'xsi:schemaLocation="http://pixinsight.com/xisf '
    'http://pixinsight.com/xisf/xisf-1.0.xsd"'
)

COMPRESSORS = {
    "zlib": lambda b: zlib.compress(b),
    "lz4": lambda b: lz4.block.compress(b, store_size=False),
    "lz4hc": lambda b: lz4.block.compress(
        b, mode="high_compression", compression=9, store_size=False
    ),
    "zstd": lambda b: zstandard.compress(b, level=3),
}


def sample_image(channels=3, width=6, height=4, dtype=np.uint16):
    """A small deterministic test image, channels-last."""
    n = height * width * channels
    return np.arange(n, dtype=dtype).reshape(height, width, channels)


def planar_bytes(im):
    """Serialize a channels-last image the way XISF stores it: planar."""
    return np.ascontiguousarray(np.transpose(im, (2, 0, 1))).tobytes()


def color_space(channels):
    return "RGB" if channels == 3 else "Gray"


def encode_block(raw, encoding="base64", codec=None, item_size=None):
    """Encode a data block, returning (text, compression_attr_or_None).

    Follows the spec syntax "codec:uncompressed_size[:item_size]", where the
    trailing item_size appears only with byte shuffling.
    """
    compression = None
    if codec:
        block = XISF._shuffle(raw, item_size) if item_size else raw
        codec_str = f"{codec}+sh" if item_size else codec
        compression = f"{codec_str}:{len(raw)}"
        if item_size:
            compression += f":{item_size}"
        raw = COMPRESSORS[codec](block)
    if encoding == "base64":
        return base64.b64encode(raw).decode("ascii"), compression
    if encoding == "hex":
        # The spec mandates lowercase hexadecimal for Base16 data.
        return raw.hex(), compression
    raise ValueError(encoding)


def image_element(im, location="inline", encoding="base64", codec=None, item_size=None):
    """Build an <Image> element with an inline or embedded data block."""
    h, w, c = im.shape
    text, compression = encode_block(
        planar_bytes(im), encoding=encoding, codec=codec, item_size=item_size
    )
    attrs = (
        f'geometry="{w}:{h}:{c}" '
        f'sampleFormat="{XISF._get_sampleFormat(np.dtype(im.dtype))}" '
        f'colorSpace="{color_space(c)}"'
    )
    if location == "inline":
        loc = f"inline:{encoding}"
        if compression:
            # For non-embedded blocks the compression attribute belongs to the
            # element that serializes the block.
            attrs += f' compression="{compression}"'
        return f'<Image {attrs} location="{loc}">{text}</Image>'
    if location == "embedded":
        data_attrs = f'encoding="{encoding}"'
        if compression:
            # For embedded blocks it belongs to the child Data element instead.
            data_attrs += f' compression="{compression}"'
        return (
            f'<Image {attrs} location="embedded">'
            f"<Data {data_attrs}>{text}</Data></Image>"
        )
    raise ValueError(location)


def vector_property_element(
    values, ptype="F64Vector", location="inline", encoding="base64",
    codec=None, item_size=None, extra_attrs="",
):
    """Build a <Property> element holding a vector in an inline/embedded block."""
    arr = np.asarray(values)
    text, compression = encode_block(
        arr.tobytes(), encoding=encoding, codec=codec, item_size=item_size
    )
    attrs = f'id="{ptype}:Test" type="{ptype}" length="{arr.size}"{extra_attrs}'
    return _property_body(attrs, text, compression, location, encoding)


def matrix_property_element(
    values, ptype="F64Matrix", location="inline", encoding="base64",
    codec=None, item_size=None, extra_attrs="",
):
    """Build a <Property> element holding a matrix in an inline/embedded block."""
    arr = np.asarray(values)
    text, compression = encode_block(
        arr.tobytes(), encoding=encoding, codec=codec, item_size=item_size
    )
    attrs = (
        f'id="{ptype}:Test" type="{ptype}" '
        f'rows="{arr.shape[0]}" columns="{arr.shape[1]}"{extra_attrs}'
    )
    return _property_body(attrs, text, compression, location, encoding)


def string_property_element(
    text, ptype="String", location="inline", encoding="base64",
    codec=None, item_size=None, extra_attrs="",
):
    """Build a <Property> element holding a String in an inline/embedded block."""
    raw = text.encode("utf-8")
    body, compression = encode_block(
        raw, encoding=encoding, codec=codec, item_size=item_size
    )
    attrs = f'id="String:Test" type="String"{extra_attrs}'
    return _property_body(attrs, body, compression, location, encoding)


def _property_body(attrs, text, compression, location, encoding):
    if location == "inline":
        if compression:
            attrs += f' compression="{compression}"'
        return f'<Property {attrs} location="inline:{encoding}">{text}</Property>'
    if location == "embedded":
        data_attrs = f'encoding="{encoding}"'
        if compression:
            data_attrs += f' compression="{compression}"'
        return (
            f'<Property {attrs} location="embedded">'
            f"<Data {data_attrs}>{text}</Data></Property>"
        )
    raise ValueError(location)


METADATA = '<Metadata><Property id="XISF:Keep" type="Int" value="1"/></Metadata>'


def write_monolithic(path, body, attached=b""):
    """Write a monolithic XISF file with a hand-built header.

    Returns the byte offset at which `attached` starts (4096-aligned, as
    XISF.write() aligns attached data blocks).
    """
    xml = f"<xisf {ROOT_ATTRS}>{body}</xisf>".encode("utf-8")
    head = (
        XISF._signature
        + len(xml).to_bytes(4, "little")
        + b"\0" * 4
        + xml
    )
    pad = (-len(head)) % XISF._block_alignment_size
    with open(path, "wb") as f:
        f.write(head)
        f.write(b"\0" * pad)
        f.write(attached)
    return len(head) + pad


def attached_image(path, im, codec=None, item_size=None):
    """Write a file whose image data is in an attached data block."""
    raw = planar_bytes(im)
    compression = None
    block = raw
    if codec:
        block = XISF._shuffle(raw, item_size) if item_size else raw
        codec_str = f"{codec}+sh" if item_size else codec
        compression = f"{codec_str}:{len(raw)}"
        if item_size:
            compression += f":{item_size}"
        block = COMPRESSORS[codec](block)
    # Two passes: the header length depends on the offset, which depends on the
    # header length. Pad with a generous margin so the second pass is stable.
    offset = write_monolithic(path, "", attached=block)
    h, w, c = im.shape
    attrs = (
        f'geometry="{w}:{h}:{c}" '
        f'sampleFormat="{XISF._get_sampleFormat(np.dtype(im.dtype))}" '
        f'colorSpace="{color_space(c)}"'
        + (f' compression="{compression}"' if compression else "")
    )
    body = f'<Image {attrs} location="attachment:{offset}:{len(block)}"/>{METADATA}'
    write_monolithic(path, body, attached=block)
    return path


@pytest.fixture
def make_file(tmp_path):
    """Factory writing a monolithic XISF file from a header body."""

    def _make(body, attached=b"", name="test.xisf"):
        path = tmp_path / name
        write_monolithic(path, body, attached=attached)
        return path

    return _make
