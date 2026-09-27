"""Tests for XISF data block locations and encodings.

Covers the 'inline', 'embedded' and 'attachment' block locations required by
the baseline decoder in section 7.2 of the XISF 1.0 specification, the base64
and hex encodings of section 10.3, and the placement of the compression
attribute required by section 10.6.
"""

import numpy as np
import pytest

from xisf import XISF

from .conftest import (
    METADATA,
    attached_image,
    image_element,
    matrix_property_element,
    sample_image,
    string_property_element,
    vector_property_element,
)

CODECS = ["zlib", "lz4", "lz4hc", "zstd"]
LOCATIONS = ["inline", "embedded"]


def image_body(location, im, **kwargs):
    return METADATA + image_element(im, location=location, **kwargs)


# --------------------------------------------------------------------------
# Image data blocks: inline and embedded
# --------------------------------------------------------------------------


@pytest.mark.parametrize("location", LOCATIONS)
@pytest.mark.parametrize("channels", [1, 3])
def test_image_inline_and_embedded_roundtrip(make_file, location, channels):
    """Images whose data is inline or embedded must decode to the exact bytes."""
    im = sample_image(channels=channels)
    path = make_file(image_body(location, im))

    x = XISF(str(path))
    decoded = x.read_image(0)

    assert decoded.shape == im.shape
    assert decoded.dtype == im.dtype
    np.testing.assert_array_equal(decoded, im)


@pytest.mark.parametrize("location", LOCATIONS)
def test_image_encodings(make_file, location):
    """Both base64 and (lowercase) hex encodings must be accepted."""
    im = sample_image(channels=3)
    for encoding in ("base64", "hex"):
        path = make_file(image_body(location, im, encoding=encoding))
        np.testing.assert_array_equal(XISF(str(path)).read_image(0), im)


@pytest.mark.parametrize("location", LOCATIONS)
@pytest.mark.parametrize("codec", CODECS)
def test_image_compressed(make_file, location, codec):
    """Compressed inline/embedded blocks must decompress to the original bytes.

    For embedded blocks the compression attribute lives on the child Data
    element, not on the element that serializes the block (spec 10.6).
    """
    im = sample_image(channels=3)
    path = make_file(image_body(location, im, codec=codec))
    np.testing.assert_array_equal(XISF(str(path)).read_image(0), im)


@pytest.mark.parametrize("location", LOCATIONS)
@pytest.mark.parametrize("codec", ["zlib", "lz4hc"])
def test_image_compressed_with_shuffle(make_file, location, codec):
    """Byte shuffling combined with compression must round-trip."""
    im = sample_image(channels=3, dtype=np.float32)
    path = make_file(
        image_body(location, im, codec=codec, item_size=im.dtype.itemsize)
    )
    np.testing.assert_array_equal(XISF(str(path)).read_image(0), im)


def test_channels_first_also_works_for_inline(make_file):
    """read_image(data_format='channels_first') must work for inline images."""
    im = sample_image(channels=3)
    path = make_file(image_body("inline", im))
    x = XISF(str(path))
    got = x.read_image(0, data_format="channels_first")
    assert got.shape == (3, im.shape[0], im.shape[1])
    np.testing.assert_array_equal(got, np.transpose(im, (2, 0, 1)))


def test_embedded_without_data_element_raises(make_file):
    """A location='embedded' block with no child Data element must be reported."""
    body = (
        METADATA
        + '<Image geometry="2:2:1" sampleFormat="UInt16" colorSpace="Gray"'
        ' location="embedded"/>'
    )
    path = make_file(body)
    with pytest.raises(ValueError, match="no child Data element"):
        XISF(str(path)).read_image(0)


# --------------------------------------------------------------------------
# Property data blocks: inline and embedded
# --------------------------------------------------------------------------


def _with_image(inner):
    """An Image with an attached (empty) block, plus a property inside it."""
    return (
        METADATA
        + '<Image geometry="2:2:1" sampleFormat="UInt16" colorSpace="Gray"'
        ' location="attachment:4096:0">'
        + inner
        + "</Image>"
    )


@pytest.mark.parametrize("location", LOCATIONS)
def test_vector_property_inline_and_embedded(make_file, location):
    """Vector properties must decode from inline and embedded blocks."""
    values = np.array([1.0, 2.0, 3.0, 4.0])
    body = _with_image(vector_property_element(values, location=location))
    path = make_file(body)

    prop = XISF(str(path)).get_images_metadata()[0]["XISFProperties"]["F64Vector:Test"]
    assert prop["dtype"] == np.dtype("float64")
    np.testing.assert_array_equal(prop["value"], values)


@pytest.mark.parametrize("location", LOCATIONS)
def test_matrix_property_inline_and_embedded(make_file, location):
    """Matrix properties must decode from inline and embedded blocks."""
    values = np.arange(6, dtype=np.float64).reshape(2, 3)
    body = _with_image(matrix_property_element(values, location=location))
    path = make_file(body)

    prop = XISF(str(path)).get_images_metadata()[0]["XISFProperties"]["F64Matrix:Test"]
    assert prop["value"].shape == (2, 3)
    np.testing.assert_array_equal(prop["value"], values)


@pytest.mark.parametrize("location", LOCATIONS)
def test_string_property_inline_and_embedded(make_file, location):
    """String properties must decode from inline and embedded blocks."""
    text = "an embedded string value"
    body = _with_image(string_property_element(text, location=location))
    path = make_file(body)

    prop = XISF(str(path)).get_images_metadata()[0]["XISFProperties"]["String:Test"]
    assert prop["value"] == text


@pytest.mark.parametrize("location", LOCATIONS)
@pytest.mark.parametrize("encoding", ["base64", "hex"])
def test_property_encodings(make_file, location, encoding):
    """Property data blocks must accept base64 and lowercase hex."""
    # 1.5/2.5/3.5 are chosen so the Base16 text contains a-f digits, which is
    # what actually exercises case-insensitive decoding.
    values = np.array([1.5, 2.5, 3.5])
    body = _with_image(
        vector_property_element(values, location=location, encoding=encoding)
    )
    path = make_file(body)
    prop = XISF(str(path)).get_images_metadata()[0]["XISFProperties"]["F64Vector:Test"]
    np.testing.assert_array_equal(prop["value"], values)


@pytest.mark.parametrize("location", LOCATIONS)
@pytest.mark.parametrize("codec", CODECS)
def test_vector_property_compressed(make_file, location, codec):
    """Compressed vector property blocks must decompress correctly."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    body = _with_image(vector_property_element(values, location=location, codec=codec))
    path = make_file(body)
    prop = XISF(str(path)).get_images_metadata()[0]["XISFProperties"]["F64Vector:Test"]
    np.testing.assert_array_equal(prop["value"], values)


@pytest.mark.parametrize("location", LOCATIONS)
def test_lowercase_hex_is_accepted(make_file, location):
    """Base16 data in the spec-mandated lowercase form must decode.

    Section 10.3 requires "a lowercase hexadecimal representation" for Base16.
    Regression test for the 'hex' mapping, which used base64.b16decode without
    casefold=True and so raised binascii.Error on every conformant file.

    The bytes are chosen so the hex text necessarily contains a-f digits;
    digit-only hex would decode even without casefolding, hiding the bug.
    """
    import base64

    raw = bytes([0x00, 0x0A, 0xAB, 0xFF, 0xDE, 0xAD])
    hex_text = raw.hex()
    assert hex_text == "000aabffdead"  # lowercase, and contains a-f
    # Guard the premise: this is exactly what casefold-less decoding rejects.
    with pytest.raises(Exception):
        base64.b16decode(hex_text)

    # 4x6x1 uint16 = 24 samples = 48 bytes; the distinctive bytes lead.
    nbytes = 4 * 6 * 1 * np.dtype(np.uint16).itemsize
    buf = raw + b"\x00" * (nbytes - len(raw))
    im = np.frombuffer(buf, dtype=np.uint16).reshape(4, 6, 1)

    path = make_file(image_body(location, im, encoding="hex"))
    decoded = XISF(str(path)).read_image(0)
    assert decoded.tobytes()[: len(raw)] == raw


def test_unknown_encoding_raises(make_file):
    """An unrecognized encoding must be reported, not silently mishandled."""
    body = (
        METADATA
        + '<Image geometry="2:2:1" sampleFormat="UInt16" colorSpace="Gray"'
        ' location="inline:rot13">AAAA</Image>'
    )
    path = make_file(body)
    with pytest.raises(NotImplementedError, match="encoding type 'rot13'"):
        XISF(str(path)).read_image(0)


# --------------------------------------------------------------------------
# Attached blocks (regression guard for the XML-threading change)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("codec", [None] + CODECS)
def test_attached_image(tmp_path, codec):
    """Attached image blocks must still decode after the refactor."""
    im = sample_image(channels=3)
    item = im.dtype.itemsize if codec else None
    path = attached_image(tmp_path / "a.xisf", im, codec=codec, item_size=item)
    np.testing.assert_array_equal(XISF(str(path)).read_image(0), im)


def test_multiple_images_stay_aligned(make_file):
    """_images_xml must stay index-aligned with _images_meta.

    A file with several images using different block locations would break if
    the parallel XML list drifted out of sync with the metadata list.
    """
    im_a = sample_image(channels=3, width=4, height=2)
    im_b = sample_image(channels=1, width=3, height=5)
    body = (
        METADATA
        + image_element(im_a, location="inline")
        + image_element(im_b, location="embedded")
    )
    path = make_file(body, name="multi.xisf")

    x = XISF(str(path))
    assert len(x.get_images_metadata()) == 2
    np.testing.assert_array_equal(x.read_image(0), im_a)
    np.testing.assert_array_equal(x.read_image(1), im_b)


# --------------------------------------------------------------------------
# Writer round-trips (regression guard)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("codec", [None] + CODECS)
@pytest.mark.parametrize("shuffle", [False, True])
def test_write_read_roundtrip(tmp_path, codec, shuffle):
    """Everything XISF.write() produces must read back pixel-identical."""
    im = sample_image(channels=3)
    path = tmp_path / f"rt_{codec}_{shuffle}.xisf"
    XISF.write(str(path), im, codec=codec, shuffle=shuffle)
    np.testing.assert_array_equal(XISF.read(str(path)), im)


def test_public_metadata_dicts_have_no_private_keys(tmp_path):
    """The fix must not leak private keys into the public metadata dicts.

    _images_xml is held on the instance precisely so that nothing extra leaks
    into what get_images_metadata() returns.
    """
    im = sample_image(channels=3)
    path = tmp_path / "clean.xisf"
    XISF.write(str(path), im)
    x = XISF(str(path))
    for key in x.get_images_metadata()[0]:
        assert not key.startswith("_"), f"private key leaked: {key}"
    for key in x.get_file_metadata():
        assert not key.startswith("_"), f"private key leaked: {key}"
