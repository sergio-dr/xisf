"""Tests for compressed data blocks divided into subblocks.

Covers the subblocks attribute of section 10.6 of the XISF 1.0 specification
and the uncompressed subblocks of section 10.6.1, introduced in Revision 2:
a subblock whose compressed and uncompressed lengths are equal stores its data
unmodified, every other subblock is decompressed, and each decompressed
subblock shall have exactly its uncompressed length.
"""

import base64
import hashlib
import zlib

import lz4.block
import numpy as np
import pytest
import zstandard

from xisf import XISF, XISFError

from .conftest import (
    COMPRESSORS,
    METADATA,
    color_space,
    planar_bytes,
    write_monolithic,
)

CODECS = ["zlib", "lz4", "lz4hc", "zstd"]
LOCATIONS = ["attachment", "inline", "embedded"]


def noisy_image(seed=1, width=64, height=48, channels=3, dtype=np.float32):
    """An image whose low-order bytes are noise, as in real pixel data.

    Byte shuffling turns such an image into byte planes that range from
    incompressible (the least significant bytes) to highly compressible.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    base = 0.1 + 0.5 * x / width + 0.2 * y / height
    im = base[..., None] + 0.05 * rng.standard_normal((height, width, channels))
    return im.astype(dtype)


def split(block, n):
    """Divide a block into n contiguous subblocks of (nearly) equal length."""
    bounds = np.linspace(0, len(block), n + 1).astype(int)
    return [block[a:b] for a, b in zip(bounds[:-1], bounds[1:])]


def compress_subblocks(pieces, codec, store_uncompressed=True):
    """Compress each subblock independently, as an encoder does (spec 10.6).

    With store_uncompressed, a subblock that compression would not make
    shorter is stored unmodified, as required by spec 10.6.1. Returns the
    stored subblocks and their (compressed, uncompressed) lengths.
    """
    stored = []
    for piece in pieces:
        compressed = COMPRESSORS[codec](piece)
        if store_uncompressed and len(compressed) >= len(piece):
            compressed = piece
        stored.append(compressed)
    return stored, [(len(s), len(p)) for s, p in zip(stored, pieces)]


def subblocks_attr(lengths):
    return ":".join(f"{c},{u}" for c, u in lengths)


def image_with_subblocks(
    path, im, codec, stored, lengths, location="attachment", item_size=None,
    uncompressed_size=None, extra_attrs="",
):
    """Write a file whose image data block is stored as the given subblocks."""
    raw_size = uncompressed_size if uncompressed_size is not None else im.nbytes
    codec_str = f"{codec}+sh" if item_size else codec
    compression = f"{codec_str}:{raw_size}" + (f":{item_size}" if item_size else "")
    block_attrs = f'compression="{compression}" subblocks="{subblocks_attr(lengths)}"'
    block = b"".join(stored)
    h, w, c = im.shape
    attrs = (
        f'geometry="{w}:{h}:{c}" '
        f'sampleFormat="{XISF._get_sampleFormat(np.dtype(im.dtype))}" '
        f'colorSpace="{color_space(c)}"{extra_attrs}'
    )
    text = base64.b64encode(block).decode("ascii")
    if location == "attachment":
        # Two passes: the offset of the attached block depends on the header length
        offset = write_monolithic(path, "", attached=block)
        body = (
            f'<Image {attrs} {block_attrs} location="attachment:{offset}:{len(block)}"/>'
            f"{METADATA}"
        )
        write_monolithic(path, body, attached=block)
    elif location == "inline":
        body = f'<Image {attrs} {block_attrs} location="inline:base64">{text}</Image>'
        write_monolithic(path, METADATA + body)
    elif location == "embedded":
        # For embedded blocks both attributes belong to the child Data element
        body = (
            f'<Image {attrs} location="embedded">'
            f'<Data encoding="base64" {block_attrs}>{text}</Data></Image>'
        )
        write_monolithic(path, METADATA + body)
    else:
        raise ValueError(location)
    return path


def shuffled_planar(im, item_size):
    raw = planar_bytes(im)
    return XISF._shuffle(raw, item_size) if item_size else raw


# --------------------------------------------------------------------------
# Compressed subblocks
# --------------------------------------------------------------------------


@pytest.mark.parametrize("location", LOCATIONS)
@pytest.mark.parametrize("codec", CODECS)
@pytest.mark.parametrize("shuffle", [False, True])
def test_compressed_subblocks(tmp_path, location, codec, shuffle):
    """Independently compressed subblocks must decode to the original image."""
    im = noisy_image()
    item_size = im.dtype.itemsize if shuffle else None
    stored, lengths = compress_subblocks(
        split(shuffled_planar(im, item_size), 4), codec, store_uncompressed=False
    )
    path = image_with_subblocks(
        tmp_path / "t.xisf", im, codec, stored, lengths, location, item_size
    )
    np.testing.assert_array_equal(XISF(str(path)).read_image(0), im)


@pytest.mark.parametrize("codec", CODECS)
def test_subblocks_are_exposed_in_metadata(tmp_path, codec):
    im = noisy_image()
    stored, lengths = compress_subblocks(split(planar_bytes(im), 3), codec, False)
    path = image_with_subblocks(tmp_path / "t.xisf", im, codec, stored, lengths)
    assert XISF(str(path)).get_images_metadata()[0]["subblocks"] == lengths


# --------------------------------------------------------------------------
# Uncompressed subblocks (spec 10.6.1)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("location", LOCATIONS)
@pytest.mark.parametrize("codec", CODECS)
def test_uncompressed_subblocks_with_byte_shuffling(tmp_path, location, codec):
    """Incompressible byte planes stored unmodified must decode correctly.

    The least significant byte planes of a shuffled noisy image cannot be
    compressed, so an encoder stores them as uncompressed subblocks, which
    hold shuffled bytes (spec 10.6.1).
    """
    im = noisy_image()
    item_size = im.dtype.itemsize
    stored, lengths = compress_subblocks(
        split(shuffled_planar(im, item_size), 8), codec
    )
    assert any(c == u for c, u in lengths), "no subblock was stored uncompressed"
    assert any(c < u for c, u in lengths), "no subblock was compressed"
    path = image_with_subblocks(
        tmp_path / "t.xisf", im, codec, stored, lengths, location, item_size
    )
    np.testing.assert_array_equal(XISF(str(path)).read_image(0), im)


@pytest.mark.parametrize("codec", CODECS)
def test_all_subblocks_uncompressed(tmp_path, codec):
    """A block whose subblocks are all stored unmodified is copied as is."""
    im = noisy_image()
    pieces = split(planar_bytes(im), 3)
    path = image_with_subblocks(
        tmp_path / "t.xisf", im, codec, pieces, [(len(p), len(p)) for p in pieces]
    )
    np.testing.assert_array_equal(XISF(str(path)).read_image(0), im)


@pytest.mark.parametrize("codec", CODECS)
def test_uncompressed_block_without_subblocks_attribute(tmp_path, codec):
    """Without a subblocks attribute a block is a single subblock (spec 10.6.1).

    A compressed block whose length equals its uncompressed size is therefore
    an uncompressed subblock, and its data is copied unmodified.
    """
    im = noisy_image()
    raw = planar_bytes(im)
    offset = write_monolithic(tmp_path / "t.xisf", "", attached=raw)
    h, w, c = im.shape
    body = (
        f'<Image geometry="{w}:{h}:{c}" sampleFormat="Float32" colorSpace="RGB"'
        f' compression="{codec}:{len(raw)}" location="attachment:{offset}:{len(raw)}"/>'
        f"{METADATA}"
    )
    write_monolithic(tmp_path / "t.xisf", body, attached=raw)
    np.testing.assert_array_equal(XISF(str(tmp_path / "t.xisf")).read_image(0), im)


@pytest.mark.parametrize("codec", ["zlib", "lz4"])
def test_subblock_longer_than_its_uncompressed_data_is_decompressed(tmp_path, codec):
    """Encoders that compress every subblock can generate subblocks longer than
    their uncompressed data, which a decoder shall decompress (spec 10.6.1)."""
    # Uniformly random bytes cannot be compressed, so every subblock expands
    im = np.random.default_rng(4).integers(0, 256, (48, 64, 1), dtype=np.uint8)
    stored, lengths = compress_subblocks(split(planar_bytes(im), 2), codec, False)
    assert all(c > u for c, u in lengths), "a subblock did not expand under compression"
    path = image_with_subblocks(tmp_path / "t.xisf", im, codec, stored, lengths)
    np.testing.assert_array_equal(XISF(str(path)).read_image(0), im)


def equal_length_zlib_stream(seed=3):
    """Find data whose zlib stream is exactly as long as the data.

    Encoders implementing earlier revisions of the specification could
    generate such compressed subblocks.
    """
    # A run of zeros between two runs of noise shortens the compressed stream
    # relative to the data as the run grows
    noise = np.random.default_rng(seed).integers(0, 256, 2000, dtype=np.uint8).tobytes()
    for k in range(1, 400):
        data = noise[:1000] + bytes(k) + noise[1000:]
        compressed = zlib.compress(data)
        if len(compressed) == len(data):
            return data, compressed
    raise AssertionError("no equal-length zlib stream found")


@pytest.mark.parametrize("with_subblocks_attribute", [False, True])
def test_legacy_zlib_subblock_of_equal_length_is_decompressed(
    tmp_path, with_subblocks_attribute
):
    """A compressed subblock with equal lengths, from an encoder implementing an
    earlier revision, is decompressed when that yields exactly its uncompressed
    length (spec 10.6.1)."""
    data, compressed = equal_length_zlib_stream()
    im = np.frombuffer(data, dtype=np.uint8).reshape(1, len(data), 1)
    path = tmp_path / "t.xisf"
    if with_subblocks_attribute:
        image_with_subblocks(path, im, "zlib", [compressed], [(len(data), len(data))])
    else:
        offset = write_monolithic(path, "", attached=compressed)
        body = (
            f'<Image geometry="{len(data)}:1:1" sampleFormat="UInt8" colorSpace="Gray"'
            f' compression="zlib:{len(data)}"'
            f' location="attachment:{offset}:{len(compressed)}"/>{METADATA}'
        )
        write_monolithic(path, body, attached=compressed)
    np.testing.assert_array_equal(XISF(str(path)).read_image(0), im)


def test_short_uncompressed_subblock_that_is_valid_lz4_is_copied(tmp_path):
    """An uncompressed subblock is copied even when it is also valid LZ4 data.

    The LZ4 block format has no integrity check, so a short uncompressed
    subblock can be a valid LZ4 stream that decompresses to exactly its own
    length, but to different bytes. PixInsight stores every subblock shorter
    than 64 bytes uncompressed, so this happens at the end of a block.
    """
    tail = bytes.fromhex("000000aa000000003300000000c2")
    decoded = lz4.block.decompress(tail, uncompressed_size=len(tail))
    assert len(decoded) == len(tail) and decoded != tail
    head = bytes(range(256)) * 16
    raw = head + tail
    im = np.frombuffer(raw, dtype=np.uint8).reshape(1, len(raw), 1)
    compressed_head = COMPRESSORS["lz4"](head)
    stored = [compressed_head, tail]
    lengths = [(len(compressed_head), len(head)), (len(tail), len(tail))]
    path = image_with_subblocks(tmp_path / "t.xisf", im, "lz4", stored, lengths)
    np.testing.assert_array_equal(XISF(str(path)).read_image(0), im)


# --------------------------------------------------------------------------
# Invalid data blocks
# --------------------------------------------------------------------------


@pytest.mark.parametrize("codec", CODECS)
def test_subblock_decompressing_to_wrong_length_is_invalid(tmp_path, codec):
    """A compressed subblock shall decompress to exactly its uncompressed
    length, and the data block is invalid otherwise (spec 10.6.1)."""
    im = noisy_image(channels=1)
    pieces = split(planar_bytes(im), 2)
    # The second subblock is compressed from fewer bytes than it declares
    short = pieces[1][:-64]
    stored = [COMPRESSORS[codec](pieces[0]), COMPRESSORS[codec](short)]
    lengths = [(len(stored[0]), len(pieces[0])), (len(stored[1]), len(pieces[1]))]
    path = image_with_subblocks(tmp_path / "t.xisf", im, codec, stored, lengths)
    with pytest.raises(XISFError, match="subblock 2 of 2"):
        XISF(str(path)).read_image(0)


@pytest.mark.parametrize("codec", CODECS)
def test_subblock_decompressing_to_more_than_its_length_is_invalid(tmp_path, codec):
    im = noisy_image(channels=1)
    pieces = split(planar_bytes(im), 2)
    stored = [COMPRESSORS[codec](pieces[0] + b"\0" * 64), COMPRESSORS[codec](pieces[1])]
    lengths = [(len(stored[0]), len(pieces[0])), (len(stored[1]), len(pieces[1]))]
    path = image_with_subblocks(tmp_path / "t.xisf", im, codec, stored, lengths)
    with pytest.raises(XISFError, match="subblock 1 of 2"):
        XISF(str(path)).read_image(0)


def test_zlib_subblock_expanding_far_beyond_its_length_is_invalid(tmp_path):
    """Decompression stops just past the uncompressed length, so a subblock
    that expands far beyond it is rejected without inflating all of it."""
    im = noisy_image(channels=1)
    pieces = split(planar_bytes(im), 2)
    bomb = zlib.compress(bytes(64 << 20))
    stored = [COMPRESSORS["zlib"](pieces[0]), bomb]
    lengths = [(len(stored[0]), len(pieces[0])), (len(bomb), len(pieces[1]))]
    path = image_with_subblocks(tmp_path / "t.xisf", im, "zlib", stored, lengths)
    with pytest.raises(XISFError, match="subblock 2 of 2 decompresses to"):
        XISF(str(path)).read_image(0)


def test_zstd_frame_declaring_another_content_size_is_invalid(tmp_path):
    """A zstd frame declaring a content size other than the uncompressed length
    of its subblock is rejected before anything is allocated for it."""
    im = noisy_image(channels=1)
    pieces = split(planar_bytes(im), 2)
    # A single-segment frame header declaring 2^40 bytes of content
    header = zstandard.ZstdCompressor().compress(b"")[:4] + bytes([0xE0]) + (1 << 40).to_bytes(8, "little")
    stored = [COMPRESSORS["zstd"](pieces[0]), header + bytes(3)]
    lengths = [(len(stored[0]), len(pieces[0])), (len(stored[1]), len(pieces[1]))]
    path = image_with_subblocks(tmp_path / "t.xisf", im, "zstd", stored, lengths)
    with pytest.raises(XISFError, match="declares a content size"):
        XISF(str(path)).read_image(0)


def test_truncated_zlib_subblock_is_invalid(tmp_path):
    im = noisy_image(channels=1)
    pieces = split(planar_bytes(im), 2)
    stored = [COMPRESSORS["zlib"](pieces[0]), COMPRESSORS["zlib"](pieces[1])[:-8]]
    lengths = [(len(stored[0]), len(pieces[0])), (len(stored[1]), len(pieces[1]))]
    path = image_with_subblocks(tmp_path / "t.xisf", im, "zlib", stored, lengths)
    with pytest.raises(XISFError, match="subblock 2 of 2"):
        XISF(str(path)).read_image(0)


def test_undecodable_compressed_subblock_is_invalid(tmp_path):
    im = noisy_image(channels=1)
    pieces = split(planar_bytes(im), 2)
    stored = [COMPRESSORS["zlib"](pieces[0]), b"not zlib data"]
    lengths = [(len(stored[0]), len(pieces[0])), (len(stored[1]), len(pieces[1]))]
    path = image_with_subblocks(tmp_path / "t.xisf", im, "zlib", stored, lengths)
    with pytest.raises(XISFError, match="cannot be decompressed"):
        XISF(str(path)).read_image(0)


def test_compressed_lengths_must_add_up_to_the_block_length(tmp_path):
    im = noisy_image(channels=1)
    stored, lengths = compress_subblocks(split(planar_bytes(im), 2), "zstd")
    lengths[0] = (lengths[0][0] + 1, lengths[0][1])
    path = image_with_subblocks(tmp_path / "t.xisf", im, "zstd", stored, lengths)
    with pytest.raises(XISFError, match="bytes of compressed data"):
        XISF(str(path)).read_image(0)


def test_uncompressed_lengths_must_add_up_to_the_uncompressed_size(tmp_path):
    im = noisy_image(channels=1)
    stored, lengths = compress_subblocks(split(planar_bytes(im), 2), "zstd")
    path = image_with_subblocks(
        tmp_path / "t.xisf", im, "zstd", stored, lengths,
        uncompressed_size=im.nbytes + 4,
    )
    with pytest.raises(XISFError, match="bytes of uncompressed data"):
        XISF(str(path)).read_image(0)


@pytest.mark.parametrize("bad", ["", "10", "10,20,30", "a,b", "10,0", "0,10", "-5,10", "10,20:"])
def test_malformed_subblocks_attribute_is_reported(tmp_path, bad):
    raw = planar_bytes(noisy_image(channels=1))
    body = (
        f'<Image geometry="64:48:1" sampleFormat="Float32" colorSpace="Gray"'
        f' compression="zlib:{len(raw)}" subblocks="{bad}" location="inline:base64">'
        f"{base64.b64encode(zlib.compress(raw)).decode('ascii')}</Image>"
    )
    path = tmp_path / "t.xisf"
    write_monolithic(path, METADATA + body)
    with pytest.raises(XISFError, match="subblock"):
        XISF(str(path))


# --------------------------------------------------------------------------
# Checksums and properties
# --------------------------------------------------------------------------


def test_checksum_covers_uncompressed_subblocks_as_stored(tmp_path):
    """The checksum of a compressed block is computed for its data as stored,
    including the data of its uncompressed subblocks (spec 10.6.1)."""
    im = noisy_image()
    item_size = im.dtype.itemsize
    stored, lengths = compress_subblocks(split(shuffled_planar(im, item_size), 8), "zstd")
    assert any(c == u for c, u in lengths)
    digest = hashlib.sha1(b"".join(stored)).hexdigest()
    path = image_with_subblocks(
        tmp_path / "t.xisf", im, "zstd", stored, lengths, item_size=item_size,
        extra_attrs=f' checksum="sha1:{digest}"',
    )
    np.testing.assert_array_equal(XISF(str(path)).read_image(0), im)

    # Altering a byte of an uncompressed subblock must fail verification
    raw_index = next(i for i, (c, u) in enumerate(lengths) if c == u)
    altered = list(stored)
    altered[raw_index] = bytes([altered[raw_index][0] ^ 0xFF]) + altered[raw_index][1:]
    path = image_with_subblocks(
        tmp_path / "bad.xisf", im, "zstd", altered, lengths, item_size=item_size,
        extra_attrs=f' checksum="sha1:{digest}"',
    )
    with pytest.raises(XISFError, match="checksum"):
        XISF(str(path)).read_image(0)


@pytest.mark.parametrize("location", ["inline", "embedded"])
def test_vector_property_with_subblocks(tmp_path, location):
    vec = np.random.default_rng(5).standard_normal(500)
    pieces = split(XISF._shuffle(vec.tobytes(), 8), 4)
    stored, lengths = compress_subblocks(pieces, "zlib")
    assert any(c == u for c, u in lengths) and any(c < u for c, u in lengths)
    text = base64.b64encode(b"".join(stored)).decode("ascii")
    block_attrs = (
        f'compression="zlib+sh:{vec.nbytes}:8" subblocks="{subblocks_attr(lengths)}"'
    )
    attrs = f'id="F64Vector:Test" type="F64Vector" length="{vec.size}"'
    if location == "inline":
        prop = f'<Property {attrs} {block_attrs} location="inline:base64">{text}</Property>'
    else:
        prop = (
            f'<Property {attrs} location="embedded">'
            f'<Data encoding="base64" {block_attrs}>{text}</Data></Property>'
        )
    path = tmp_path / "t.xisf"
    write_monolithic(path, f"<Metadata>{prop}</Metadata>")
    value = XISF(str(path)).get_file_metadata()["F64Vector:Test"]["value"]
    np.testing.assert_array_equal(value, vec)


def test_attached_vector_property_with_subblocks(tmp_path):
    vec = np.random.default_rng(6).standard_normal(500)
    stored, lengths = compress_subblocks(split(XISF._shuffle(vec.tobytes(), 8), 4), "zstd")
    assert any(c == u for c, u in lengths) and any(c < u for c, u in lengths)
    block = b"".join(stored)
    block_attrs = (
        f'compression="zstd+sh:{vec.nbytes}:8" subblocks="{subblocks_attr(lengths)}"'
    )

    def body(offset):
        return (
            f'<Metadata><Property id="F64Vector:Test" type="F64Vector"'
            f' length="{vec.size}" {block_attrs}'
            f' location="attachment:{offset}:{len(block)}"/></Metadata>'
        )

    path = tmp_path / "t.xisf"
    # Two passes: the offset of the attached block depends on the header length
    offset = write_monolithic(path, body(0), attached=block)
    assert write_monolithic(path, body(offset), attached=block) == offset
    value = XISF(str(path)).get_file_metadata()["F64Vector:Test"]["value"]
    np.testing.assert_array_equal(value, vec)


def test_string_property_with_subblocks(tmp_path):
    text = "".join(chr(0x41 + i % 26) for i in range(3000)) + "end"
    raw = text.encode("utf-8")
    pieces = split(raw, 3)
    # The last subblock is stored uncompressed even though it is compressible,
    # as an encoder may do for subblocks too short to be worth compressing
    stored = [zlib.compress(pieces[0]), zlib.compress(pieces[1]), pieces[2]]
    lengths = [(len(s), len(p)) for s, p in zip(stored, pieces)]
    prop = (
        f'<Property id="String:Test" type="String" compression="zlib:{len(raw)}"'
        f' subblocks="{subblocks_attr(lengths)}" location="inline:base64">'
        f'{base64.b64encode(b"".join(stored)).decode("ascii")}</Property>'
    )
    path = tmp_path / "t.xisf"
    write_monolithic(path, f"<Metadata>{prop}</Metadata>")
    assert XISF(str(path)).get_file_metadata()["String:Test"]["value"] == text
