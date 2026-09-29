"""Tests for XISF data block checksums.

Covers the checksum attribute of section 10.5 of the XISF 1.0 specification, the
digest of a compressed block of section 10.6.1, and the XISF:ChecksumAlgorithms
file property of section 11.2. A baseline decoder shall verify SHA-1, SHA-256 and
SHA-512 checksums (section 7.2), and every decoder shall verify every data block
checksum it finds.
"""

import base64
import hashlib
import re

import numpy as np
import pytest

from xisf import XISF, XISFError, XISFWarning

from .conftest import METADATA, image_element

# The five algorithms of spec 10.5 Table 9, as canonical name -> (constructor
# name, alternate spelling, digest length in hex characters)
ALGORITHMS = {
    "sha-1": ("sha1", "sha1", 40),
    "sha-256": ("sha256", "sha256", 64),
    "sha-512": ("sha512", "sha512", 128),
    "sha3-256": ("sha3_256", None, 64),
    "sha3-512": ("sha3_512", None, 128),
}

# The three algorithms every decoder shall support (spec 7.2)
MANDATORY = ("sha-1", "sha-256", "sha-512")


def smooth_image(seed=8, side=64):
    """An image every codec can actually compress.

    A writer that gains nothing from compression discards the compressed block
    and writes the data uncompressed (see _serialize_data_block), so a test that
    needs a compressed block cannot use random data: a codec that does not
    reduce the size leaves no compression attribute behind.
    """
    ramp = np.linspace(0, 1, side, dtype=np.float32)
    return np.stack(np.meshgrid(ramp, ramp, ramp[:side], indexing="ij"), axis=-1)


def build_file(path, im, image_attrs="", property_xml=None, image_text=None):
    """Write an XISF file with a hand-built header and no attached blocks.

    property_xml replaces the contents of the Metadata element, so a test can
    declare exactly the properties it needs.
    """
    if image_text is None:
        image_text = base64.b64encode(im.tobytes()).decode()
    metadata = METADATA if property_xml is None else f"<Metadata>{property_xml}</Metadata>"
    xml = (
        f"<xisf xmlns='http://www.pixinsight.com/xisf' version='1.0'>"
        f'<Image geometry="6:2:1" sampleFormat="UInt8" colorSpace="Gray"'
        f' location="inline:base64"{image_attrs}>{image_text}</Image>'
        f"{metadata}"
        f"</xisf>"
    ).encode()
    head = (
        XISF._signature
        + len(xml).to_bytes(4, "little")
        + (0).to_bytes(4, "little")
        + xml
    )
    head += b"\0" * ((4096 - len(head) % 4096) % 4096)
    path.write_bytes(head)
    return path


def header_of(path):
    raw = path.read_bytes()
    length = int.from_bytes(raw[8:12], "little")
    return raw[16 : 16 + length].rstrip(b"\0").decode()


def sha1(data):
    return "sha-1:" + hashlib.sha1(data).hexdigest()


# ---------------------------------------------------------------- decoding


@pytest.mark.parametrize("algorithm", MANDATORY)
def test_mandatory_algorithms_are_verified(tmp_path, algorithm):
    """A baseline decoder shall verify SHA-1, SHA-256 and SHA-512 (spec 7.2)."""
    im = np.arange(12, dtype=np.uint8).reshape(2, 6)
    digest = f"{algorithm}:{hashlib.new(ALGORITHMS[algorithm][0], im.tobytes()).hexdigest()}"
    path = build_file(tmp_path / "ok.xisf", im, image_attrs=f' checksum="{digest}"')

    x = XISF(str(path))
    assert x.get_images_metadata()[0]["checksum"] == digest
    np.testing.assert_array_equal(x.read_image(0)[..., 0], im)


def test_absent_checksum_is_accepted(tmp_path):
    """The attribute is optional for a non-signed unit (spec 10.5)."""
    im = np.arange(12, dtype=np.uint8).reshape(2, 6)
    path = build_file(tmp_path / "none.xisf", im)
    x = XISF(str(path))
    assert "checksum" not in x.get_images_metadata()[0]
    np.testing.assert_array_equal(x.read_image(0)[..., 0], im)


def test_mismatched_checksum_raises(tmp_path):
    """A failed verification shall not make the block available (spec 10.5)."""
    im = np.arange(12, dtype=np.uint8).reshape(2, 6)
    path = build_file(
        tmp_path / "bad.xisf", im, image_attrs=f' checksum="{sha1(b"wrong")}"'
    )
    with pytest.raises(XISFError, match="fails checksum verification"):
        XISF(str(path)).read_image(0)


def test_verify_checksums_false_warns_and_returns(tmp_path):
    """The opt-out downgrades the failure to a warning."""
    im = np.arange(12, dtype=np.uint8).reshape(2, 6)
    path = build_file(
        tmp_path / "bad.xisf", im, image_attrs=f' checksum="{sha1(b"wrong")}"'
    )
    with pytest.warns(XISFWarning, match="fails checksum verification"):
        back = XISF(str(path), verify_checksums=False).read_image(0)
    np.testing.assert_array_equal(back[..., 0], im)


def test_verify_checksums_must_be_bool(tmp_path):
    im = np.arange(12, dtype=np.uint8).reshape(2, 6)
    path = build_file(tmp_path / "ok.xisf", im)
    with pytest.raises(XISFError, match="verify_checksums"):
        XISF(str(path), verify_checksums="yes")


def test_uppercase_digest_is_accepted(tmp_path):
    """Digests are lowercase in the spec, but a decoder need not be pedantic."""
    im = np.arange(12, dtype=np.uint8).reshape(2, 6)
    digest = sha1(im.tobytes()).upper().replace("SHA-1:", "sha-1:")
    path = build_file(tmp_path / "up.xisf", im, image_attrs=f' checksum="{digest}"')
    np.testing.assert_array_equal(XISF(str(path)).read_image(0)[..., 0], im)


def test_alternate_algorithm_spelling_is_accepted(tmp_path):
    """Table 9 lists alternate spellings, which shall be accepted too."""
    im = np.arange(12, dtype=np.uint8).reshape(2, 6)
    digest = f"sha1:{hashlib.sha1(im.tobytes()).hexdigest()}"
    path = build_file(tmp_path / "alt.xisf", im, image_attrs=f' checksum="{digest}"')
    np.testing.assert_array_equal(XISF(str(path)).read_image(0)[..., 0], im)


@pytest.mark.parametrize("bad", ["sha-1", ":abc", "sha-1:", "algo:"])
def test_malformed_checksum_is_reported(tmp_path, bad):
    """The syntax is 'algorithm:digest' (spec 10.5)."""
    im = np.arange(12, dtype=np.uint8).reshape(2, 6)
    path = build_file(
        tmp_path / "mal.xisf", im, image_attrs=f' checksum="{bad}"'
    )
    with pytest.raises(XISFError, match="checksum"):
        XISF(str(path)).read_image(0)


def test_unknown_algorithm_is_reported(tmp_path):
    """Only the algorithms of Table 9 are defined."""
    im = np.arange(12, dtype=np.uint8).reshape(2, 6)
    path = build_file(
        tmp_path / "unk.xisf", im, image_attrs=' checksum="md5:d41d8cd98f00b204e9800998ecf8427e"'
    )
    with pytest.raises(XISFError, match="unknown checksum algorithm"):
        XISF(str(path)).read_image(0)


def test_altered_attached_block_is_detected(tmp_path):
    """The whole point: a flipped bit in an attached block is caught."""
    im = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32)
    path = tmp_path / "att.xisf"
    XISF.write(str(path), im, "t")

    raw = bytearray(path.read_bytes())
    header = header_of(path)
    pos, size = map(
        int, re.search(r'location="attachment:(\d+):(\d+)"', header).groups()
    )
    raw[pos] ^= 0xFF
    (tmp_path / "corrupt.xisf").write_bytes(bytes(raw))

    with pytest.raises(XISFError, match="fails checksum verification"):
        XISF(str(tmp_path / "corrupt.xisf")).read_image(0)


def test_altered_inline_block_is_detected(tmp_path):
    """A flipped bit in the decoded inline data is caught."""
    im = np.arange(12, dtype=np.uint8).reshape(2, 6)
    digest = sha1(im.tobytes())
    good = build_file(tmp_path / "good.xisf", im, image_attrs=f' checksum="{digest}"')
    header = header_of(good)
    text = re.search(r'location="inline:base64"[^>]*>([^<]*)<', header).group(1)
    altered = bytearray(base64.b64decode(text))
    altered[0] ^= 0xFF
    path = build_file(
        tmp_path / "bad.xisf",
        im,
        image_attrs=f' checksum="{digest}"',
        image_text=base64.b64encode(bytes(altered)).decode(),
    )
    with pytest.raises(XISFError, match="fails checksum verification"):
        XISF(str(path)).read_image(0)


# ------------------------------------------- digest is over the right bytes


def test_inline_digest_is_over_the_decoded_data(tmp_path):
    """The digest is for the decoded binary data, not its Base64 text (spec 10.5)."""
    im = np.arange(12, dtype=np.uint8).reshape(2, 6)
    text = base64.b64encode(im.tobytes()).decode()
    path = build_file(
        tmp_path / "inl.xisf", im, image_attrs=f' checksum="{sha1(im.tobytes())}"'
    )
    np.testing.assert_array_equal(XISF(str(path)).read_image(0)[..., 0], im)

    # The digest of the Base64 text itself must be rejected
    path = build_file(
        tmp_path / "txt.xisf",
        im,
        image_attrs=f' checksum="{sha1(text.encode())}"',
    )
    with pytest.raises(XISFError, match="fails checksum verification"):
        XISF(str(path)).read_image(0)


def test_compressed_digest_is_over_the_compressed_data(tmp_path):
    """The digest of a compressed block is for the compressed data (spec 10.6.1)."""
    im = np.random.default_rng(1).random((256, 256, 3)).astype(np.float32)
    path = tmp_path / "comp.xisf"
    _, codec = XISF.write(str(path), im, "t", codec="zstd")
    assert codec == "zstd", "the block must actually be compressed for this test"

    raw = path.read_bytes()
    header = header_of(path)
    pos, size = map(
        int, re.search(r'location="attachment:(\d+):(\d+)"', header).groups()
    )
    compressed = raw[pos : pos + size]
    assert len(compressed) < im.nbytes

    # The declared digest matches the compressed bytes, not the original ones
    declared = re.search(r'checksum="([^"]+)"', header).group(1)
    assert declared == sha1(compressed)
    assert declared != sha1(im.tobytes())

    # And the file reads back correctly
    np.testing.assert_array_equal(XISF(str(path)).read_image(0), im)


def test_compressed_block_is_not_decompressed_after_failure(tmp_path):
    """A decoder shall not decompress altered compressed data (spec 10.6.1)."""
    im = np.random.default_rng(2).random((256, 256, 3)).astype(np.float32)
    path = tmp_path / "comp.xisf"
    XISF.write(str(path), im, "t", codec="zstd")

    raw = bytearray(path.read_bytes())
    header = header_of(path)
    pos, _ = map(int, re.search(r'location="attachment:(\d+):(\d+)"', header).groups())
    raw[pos] ^= 0xFF
    (tmp_path / "corrupt.xisf").write_bytes(bytes(raw))

    # A decompression error would mean the altered bytes reached the codec, so
    # the exception has to be the checksum failure and not a codec error. The
    # message is checked for the same reason: the codecs disagree about what a
    # corrupt block does, and a broad exception type would not tell them apart.
    with pytest.raises(XISFError) as exc:
        XISF(str(tmp_path / "corrupt.xisf")).read_image(0)
    assert "fails checksum verification" in str(exc.value)


def test_compressed_block_verifies_with_every_codec(tmp_path):
    """byteOrder and checksum compose, for every codec (spec 10.4, 10.6.1)."""
    im = np.random.default_rng(3).random((128, 128, 3)).astype(np.float32)
    for codec in ("zlib", "lz4", "lz4hc", "zstd"):
        path = tmp_path / f"{codec}.xisf"
        XISF.write(str(path), im, "t", codec=codec, shuffle=True)
        np.testing.assert_array_equal(
            XISF(str(path)).read_image(0), im, err_msg=codec
        )


# ---------------------------------------------------------------- properties


def test_property_data_block_is_verified(tmp_path):
    """Verification covers property data blocks, not just image blocks."""
    vec = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float64)
    text = base64.b64encode(vec.tobytes()).decode()
    good = (
        f'<Property id="V" type="F64Vector" length="4"'
        f' location="inline:base64" checksum="{sha1(vec.tobytes())}">{text}</Property>'
    )
    path = build_file(
        tmp_path / "p.xisf", np.arange(12, dtype=np.uint8).reshape(2, 6), property_xml=good
    )
    props = XISF(str(path)).get_file_metadata()
    np.testing.assert_array_equal(props["V"]["value"], vec)

    # An altered property block is caught
    bad = good.replace(sha1(vec.tobytes()), sha1(b"nope"))
    path = build_file(
        tmp_path / "pb.xisf", np.arange(12, dtype=np.uint8).reshape(2, 6), property_xml=bad
    )
    with pytest.raises(XISFError, match="fails checksum verification"):
        XISF(str(path)).get_file_metadata()


# ----------------------------------------------------------------- encoding


def test_writer_emits_sha1_by_default(tmp_path):
    """The default is SHA-1, the algorithm recommended by the spec."""
    im = np.arange(2 * 3, dtype=np.uint16).reshape(2, 3)
    path = tmp_path / "d.xisf"
    XISF.write(str(path), im, "t")

    header = header_of(path)
    declared = re.search(r'checksum="([^"]+)"', header).group(1)
    # The digest covers the serialized block, which for a single-channel image is
    # the array itself
    assert declared == sha1(im.tobytes())
    assert (
        XISF(str(path)).get_file_metadata()["XISF:ChecksumAlgorithms"]["value"]
        == "sha-1"
    )
    np.testing.assert_array_equal(
        XISF(str(path)).read_image(0, data_format="channels_first")[0], im
    )


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_writer_supports_every_algorithm(tmp_path, algorithm):
    constructor, _, hexlen = ALGORITHMS[algorithm]
    im = np.arange(2 * 3, dtype=np.uint16).reshape(2, 3)
    path = tmp_path / f"{algorithm}.xisf"
    XISF.write(str(path), im, "t", checksum=algorithm)

    declared = re.search(r'checksum="([^"]+)"', header_of(path)).group(1)
    assert declared.startswith(f"{algorithm}:")
    assert len(declared.partition(":")[2]) == hexlen
    assert declared.partition(":")[2] == hashlib.new(constructor, im.tobytes()).hexdigest()
    np.testing.assert_array_equal(
        XISF(str(path)).read_image(0, data_format="channels_first")[0], im
    )


@pytest.mark.parametrize("spelling", ["sha-1", "sha1"])
def test_writer_accepts_the_alternate_spelling(tmp_path, spelling):
    """The alternate spellings of Table 9 are normalized to the canonical value."""
    im = np.arange(2 * 3, dtype=np.uint16).reshape(2, 3)
    path = tmp_path / "alt.xisf"
    XISF.write(str(path), im, "t", checksum=spelling)
    assert re.search(r'checksum="(sha-1):', header_of(path))


def test_writer_accepts_true(tmp_path):
    im = np.arange(2 * 3, dtype=np.uint16).reshape(2, 3)
    path = tmp_path / "t.xisf"
    XISF.write(str(path), im, "t", checksum=True)
    assert re.search(r'checksum="sha-1:', header_of(path))


@pytest.mark.parametrize("off", [False, None])
def test_checksum_false_writes_nothing(tmp_path, off):
    im = np.arange(2 * 3, dtype=np.uint16).reshape(2, 3)
    path = tmp_path / "off.xisf"
    XISF.write(str(path), im, "t", checksum=off)
    header = header_of(path)
    assert "checksum" not in header
    assert "XISF:ChecksumAlgorithms" not in header
    np.testing.assert_array_equal(
        XISF(str(path)).read_image(0, data_format="channels_first")[0], im
    )


def test_writer_rejects_unknown_algorithm(tmp_path):
    im = np.arange(2 * 3, dtype=np.uint16).reshape(2, 3)
    with pytest.raises(XISFError, match="unknown checksum algorithm"):
        XISF.write(str(tmp_path / "x.xisf"), im, "t", checksum="md5")
    with pytest.raises(XISFError, match="checksum must be"):
        XISF.write(str(tmp_path / "x.xisf"), im, "t", checksum=3)


def test_writer_checksums_every_data_block(tmp_path):
    """Every element that serializes a data block gets a checksum."""
    im = np.arange(2 * 3, dtype=np.uint16).reshape(2, 3)
    path = tmp_path / "multi.xisf"
    XISF.write(
        str(path),
        im,
        "t",
        image_metadata={
            "XISFProperties": {
                "Big": {
                    "id": "Big",
                    "type": "F64Vector",
                    "value": np.arange(4000, dtype=np.float64),
                }
            }
        },
    )
    header = header_of(path)
    # the image block plus the attached property block
    assert len(re.findall(r'checksum="', header)) == 2
    # and the file still reads back
    x = XISF(str(path))
    np.testing.assert_array_equal(
        x.read_image(0, data_format="channels_first")[0], im
    )
    np.testing.assert_array_equal(
        x.get_images_metadata()[0]["XISFProperties"]["Big"]["value"],
        np.arange(4000, dtype=np.float64),
    )


def test_checksum_algorithms_property_lists_algorithms(tmp_path):
    """The applied algorithms are enumerated in a file property (spec 11.2)."""
    im = np.arange(2 * 3, dtype=np.uint16).reshape(2, 3)
    path = tmp_path / "list.xisf"
    XISF.write(str(path), im, "t", checksum="sha-256")
    value = XISF(str(path)).get_file_metadata()["XISF:ChecksumAlgorithms"]["value"]
    assert value == "sha-256"


def test_checksum_and_compression_compose(tmp_path):
    """A compressed file carries a checksum of the compressed block."""
    im = np.random.default_rng(4).random((128, 128, 3)).astype(np.float32)
    path = tmp_path / "zc.xisf"
    XISF.write(str(path), im, "t", codec="zstd", shuffle=True, checksum="sha-256")
    header = header_of(path)
    assert 'compression="zstd' in header
    raw = path.read_bytes()
    pos, size = map(
        int, re.search(r'location="attachment:(\d+):(\d+)"', header).groups()
    )
    compressed = raw[pos : pos + size]
    declared = re.search(r'checksum="([^"]+)"', header).group(1)
    assert declared == "sha-256:" + hashlib.sha256(compressed).hexdigest()
    np.testing.assert_array_equal(XISF(str(path)).read_image(0), im)


def test_write_read_round_trip_is_lossless(tmp_path):
    """End to end: writing and reading an image preserves it exactly."""
    im = np.random.default_rng(5).random((20, 30, 3)).astype(np.float32)
    path = tmp_path / "rt.xisf"
    XISF.write(str(path), im, "t", pixel_storage="normal")
    np.testing.assert_array_equal(XISF(str(path)).read_image(0), im)


# --------------------------------------------------------------------------
# A failed verification of a compressed block is always fatal
# --------------------------------------------------------------------------


def test_verify_checksums_false_still_refuses_a_corrupt_compressed_block(tmp_path):
    """The opt-out does not extend to a compressed block.

    Decompressing bytes that are known to be altered returns wrong pixels
    without reporting anything, since lz4 and zstd decompress a corrupt block
    silently, so a mismatch on a compressed block raises whatever the flag says.
    """
    im = smooth_image()
    path = tmp_path / "comp.xisf"
    XISF.write(str(path), im, "t", codec="lz4", checksum="sha-256")
    assert "compression" in header_of(path), "the block must actually be compressed"

    raw = bytearray(path.read_bytes())
    pos, _ = map(int, re.search(r'location="attachment:(\d+):(\d+)"', header_of(path)).groups())
    raw[pos] ^= 0xFF
    corrupt = tmp_path / "corrupt.xisf"
    corrupt.write_bytes(bytes(raw))

    with pytest.raises(XISFError) as exc:
        XISF(str(corrupt), verify_checksums=False).read_image(0)
    message = str(exc.value)
    assert "fails checksum verification" in message
    assert "compressed" in message
    assert "verify_checksums=False" in message


@pytest.mark.parametrize("codec", ["zlib", "lz4", "lz4hc", "zstd"])
def test_every_codec_is_refused_after_a_failed_verification(tmp_path, codec):
    """No codec is allowed to decompress a block that failed verification.

    The codecs do not agree on what a corrupt block does: zlib raises a
    decompression error, and lz4 and zstd return silently wrong bytes. Refusing
    before the codec is reached makes the outcome the same for all of them.
    """
    im = smooth_image()
    path = tmp_path / f"{codec}.xisf"
    XISF.write(str(path), im, "t", codec=codec)
    # A codec that cannot reduce the size is not used, and then the block is not
    # compressed, so this test would not be testing what it claims to
    assert "compression" in header_of(path), f"{codec} did not compress the block"

    raw = bytearray(path.read_bytes())
    pos, _ = map(int, re.search(r'location="attachment:(\d+):(\d+)"', header_of(path)).groups())
    raw[pos + 4] ^= 0xFF
    corrupt = tmp_path / f"corrupt-{codec}.xisf"
    corrupt.write_bytes(bytes(raw))

    with pytest.raises(XISFError, match="fails checksum verification"):
        XISF(str(corrupt), verify_checksums=False).read_image(0)


def test_uncompressed_block_is_still_returned_with_the_opt_out(tmp_path):
    """The opt-out keeps working for a block that is not compressed."""
    im = np.arange(12, dtype=np.uint8).reshape(2, 6)
    path = build_file(
        tmp_path / "bad.xisf", im, image_attrs=f' checksum="{sha1(b"wrong")}"'
    )
    with pytest.warns(XISFWarning, match="fails checksum verification"):
        back = XISF(str(path), verify_checksums=False).read_image(0)
    np.testing.assert_array_equal(back[..., 0], im)
