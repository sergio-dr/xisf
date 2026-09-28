"""Tests for image geometry, pixel storage and color space in the writer.

Relevant spec sections:

  7.1     a baseline encoder shall write pixel data in the planar model
  7.2     a baseline decoder shall read both the planar and normal models
  8.5.3.1 planar: each channel is a contiguous run of samples, channels in order
  8.5.3.2 normal: the samples of each pixel are stored together
  11.5.1  geometry is dim1:...:dimN:channel-count, dim1 being the X-axis
  11.5.2  pixelStorage defaults to planar; colorSpace is Gray, RGB or Lab
"""

import struct
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from xisf import XISF, XISFError

from .conftest import METADATA, image_element, normal_bytes, planar_bytes, write_monolithic


def read_image_attrs(path):
    """Return the Image attributes actually serialized into the file."""
    with open(path, "rb") as f:
        assert f.read(8) == b"XISF0100"
        n = struct.unpack("<I", f.read(4))[0]
        f.read(4)  # reserved
        root = ET.fromstring(f.read(n).decode())
    image = [e for e in root.iter() if e.tag.endswith("Image")][0]
    return dict(image.attrib)


# --------------------------------------------------------------------------
# 2-D input: the natural numpy form of a grayscale image
# --------------------------------------------------------------------------


def test_write_2d_grayscale(tmp_path):
    """A 2-D array is a single-channel image, not an error.

    Regression test: the writer read im_data.shape[2] unconditionally, so a 2-D
    array raised IndexError: tuple index out of range.
    """
    im = np.arange(24, dtype=np.uint16).reshape(4, 6)
    path = tmp_path / "g.xisf"
    XISF.write(str(path), im)

    attrs = read_image_attrs(path)
    # geometry is width:height:channel-count (spec 11.5.1)
    assert attrs["geometry"] == "6:4:1"
    assert attrs["colorSpace"] == "Gray"

    back = XISF(str(path)).read_image(0)
    assert back.shape == (4, 6, 1)
    np.testing.assert_array_equal(back[..., 0], im)


def test_zero_dimensional_input_is_reported(tmp_path):
    """A 0-D array has no dimension, but geometry requires N >= 1.

    Regression test: 1-D raised IndexError and 4-D raised a TypeError from
    '%d:%d:%d' % geometry, neither of which named the offending input.
    """
    path = tmp_path / "bad.xisf"
    with pytest.raises(XISFError, match="at least one dimension"):
        XISF.write(str(path), np.zeros((), dtype=np.uint16))


def test_zero_length_dimension_is_reported(tmp_path):
    """Every dimi item shall be greater than zero (spec 11.5.1)."""
    path = tmp_path / "bad.xisf"
    with pytest.raises(XISFError, match="greater than zero"):
        XISF.write(str(path), np.zeros((0, 4), dtype=np.uint16))


def test_non_ndarray_is_reported(tmp_path):
    path = tmp_path / "bad.xisf"
    with pytest.raises(XISFError, match="ndarray"):
        XISF.write(str(path), [[[1, 2]], [[3, 4]]])


# --------------------------------------------------------------------------
# geometry must agree with the data block
# --------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(4, 6, 1), (6, 4, 1), (4, 6, 3), (1, 1, 1)])
def test_channels_last_roundtrip(tmp_path, shape):
    """(height, width, channels) input round-trips to the identical array."""
    im = np.arange(int(np.prod(shape)), dtype=np.uint16).reshape(shape)
    path = tmp_path / "hwc.xisf"
    XISF.write(str(path), im)

    h, w, c = shape
    assert read_image_attrs(path)["geometry"] == f"{w}:{h}:{c}"
    np.testing.assert_array_equal(XISF(str(path)).read_image(0), im)


@pytest.mark.parametrize("shape", [(3, 4, 6), (1, 4, 6), (3, 2, 6)])
def test_inferred_planar_roundtrip(tmp_path, shape):
    """A (channels, height, width) array is recognized and written correctly.

    Regression test: the writer set geometry = im_data.shape, so (3,4,6) was
    serialized as geometry="3:4:6", i.e. width 3, height 4 and six channels,
    while the bytes held three channels of 4x6. Reading that back gave a
    (4,3,6) array with scrambled pixels and no error of any kind.
    """
    chw = np.arange(int(np.prod(shape)), dtype=np.uint16).reshape(shape)
    path = tmp_path / "chw.xisf"
    XISF.write(str(path), chw)

    c, h, w = shape
    # a trailing axis that is neither 1 nor 3 is what selects the planar model
    assert w not in (1, 3)
    assert read_image_attrs(path)["geometry"] == f"{w}:{h}:{c}"
    # read_image returns (height, width, channels)
    np.testing.assert_array_equal(XISF(str(path)).read_image(0), chw.transpose(1, 2, 0))


@pytest.mark.parametrize("shape", [(3, 4, 6), (1, 4, 6), (3, 2, 6)])
def test_explicit_pixel_storage_planar(tmp_path, shape):
    """An explicit 'planar' reproduces the inferred behaviour for keras input."""
    chw = np.arange(int(np.prod(shape)), dtype=np.uint16).reshape(shape)
    path = tmp_path / "ps.xisf"
    XISF.write(str(path), chw, pixel_storage="planar")
    c, h, w = shape
    assert read_image_attrs(path)["geometry"] == f"{w}:{h}:{c}"
    np.testing.assert_array_equal(
        XISF(str(path)).read_image(0), chw.transpose(1, 2, 0)
    )


@pytest.mark.parametrize("shape", [(4, 6, 3), (6, 3, 1)])
def test_explicit_pixel_storage_normal(tmp_path, shape):
    """An explicit 'normal' writes a (height, width, channels) array as given."""
    hwc = np.arange(int(np.prod(shape)), dtype=np.uint16).reshape(shape)
    path = tmp_path / "pn.xisf"
    XISF.write(str(path), hwc, pixel_storage="normal")
    h, w, c = shape
    assert read_image_attrs(path)["geometry"] == f"{w}:{h}:{c}"
    np.testing.assert_array_equal(XISF(str(path)).read_image(0), hwc)


def test_pixel_storage_is_case_insensitive(tmp_path):
    chw = np.arange(3 * 4 * 6, dtype=np.uint16).reshape(3, 4, 6)
    for spelling in ("Planar", "PLANAR", "planar"):
        XISF.write(str(tmp_path / "c.xisf"), chw, pixel_storage=spelling)
    hwc = np.arange(4 * 6 * 3, dtype=np.uint16).reshape(4, 6, 3)
    for spelling in ("Normal", "NORMAL", "normal"):
        XISF.write(str(tmp_path / "c.xisf"), hwc, pixel_storage=spelling)


@pytest.mark.parametrize("value", ["nope", "channels_last", "channels_first", 1])
def test_invalid_pixel_storage_is_reported(tmp_path, value):
    """The parameter uses spec naming, so keras names are rejected loudly."""
    with pytest.raises(XISFError, match="must be 'planar' or 'normal'"):
        XISF.write(
            str(tmp_path / "b.xisf"),
            np.zeros((2, 3, 4), dtype=np.uint16),
            pixel_storage=value,
        )


def test_docstring_explains_keras_layouts():
    """The docstring must map the spec names onto the keras/numpy conventions."""
    doc = XISF.write.__doc__
    assert "'planar'" in doc and "'normal'" in doc
    assert "channels first" in doc and "channels last" in doc
    assert "keras" in doc.lower()


# --------------------------------------------------------------------------
# colorSpace must name a real color space
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape,expected", [((4, 6, 1), "Gray"), ((4, 6, 3), "RGB")]
)
def test_color_space_for_valid_channel_counts(tmp_path, shape, expected):
    path = tmp_path / "cs.xisf"
    XISF.write(str(path), np.zeros(shape, dtype=np.uint16))
    assert read_image_attrs(path)["colorSpace"] == expected


@pytest.mark.parametrize("channels", [2, 4, 5, 6])
@pytest.mark.parametrize("storage", ["normal", "planar"])
def test_unsupported_channel_count_is_reported(tmp_path, channels, storage):
    """Table 14 defines color spaces for one or three channels only.

    Regression test: any channel count other than 1 was labelled "RGB". The
    channel axis is named explicitly here, since a 3-D array whose axes are
    none of 1 or 3 is otherwise read as three single-channel dimensions.
    """
    shape = (4, 6, channels) if storage == "normal" else (channels, 4, 6)
    with pytest.raises(XISFError, match="Table 14"):
        XISF.write(
            str(tmp_path / "cs.xisf"),
            np.zeros(shape, dtype=np.uint16),
            pixel_storage=storage,
        )


# --------------------------------------------------------------------------
# Pixel storage model
# --------------------------------------------------------------------------


def test_writer_declares_and_emits_planar_storage(tmp_path):
    """A baseline encoder shall write the planar model (spec 7.1)."""
    im = np.arange(4 * 6 * 3, dtype=np.uint16).reshape(4, 6, 3)
    path = tmp_path / "p.xisf"
    XISF.write(str(path), im)

    assert read_image_attrs(path)["pixelStorage"] == "Planar"

    # the data block itself must be channel by channel
    raw = path.read_bytes()
    block = raw[len(raw) - im.nbytes :]
    assert block == planar_bytes(im)
    assert block != normal_bytes(im)


def test_reader_defaults_to_planar_when_attribute_absent(tmp_path):
    """An Image without pixelStorage is planar per spec 11.5.2."""
    im = np.arange(4 * 6 * 3, dtype=np.uint16).reshape(4, 6, 3)
    path = tmp_path / "np.xisf"
    write_monolithic(path, METADATA + image_element(im, pixel_storage=None))

    assert "pixelStorage" not in read_image_attrs(path)
    np.testing.assert_array_equal(XISF(str(path)).read_image(0), im)


@pytest.mark.parametrize("storage", ["Planar", "Normal"])
def test_reader_honours_both_storage_models(tmp_path, storage):
    """A baseline decoder shall read both models (spec 7.2)."""
    im = np.arange(4 * 6 * 3, dtype=np.uint16).reshape(4, 6, 3)
    path = tmp_path / f"s{storage}.xisf"
    write_monolithic(
        path, METADATA + image_element(im, pixel_storage=storage)
    )

    assert read_image_attrs(path)["pixelStorage"] == storage
    np.testing.assert_array_equal(XISF(str(path)).read_image(0), im)


def test_misreading_normal_as_planar_would_be_wrong():
    """The two models are genuinely different, so the attribute matters.

    Guards against a 'fix' that hard-codes one reshape for both models.
    """
    shape = (2, 3, 3)
    im = np.arange(int(np.prod(shape)), dtype=np.uint32).reshape(shape)
    h, w, c = shape
    assert planar_bytes(im) != normal_bytes(im)

    # applying the planar rule to a normal-stored block corrupts the image
    misread = np.frombuffer(normal_bytes(im), dtype=np.uint32).reshape(c, h, w)
    assert not np.array_equal(misread.transpose(1, 2, 0), im)

    # and the two models coincide for a single channel, which is why 2-D input
    # is unambiguous
    gray = np.arange(6, dtype=np.uint32).reshape(2, 3, 1)
    assert planar_bytes(gray) == normal_bytes(gray)


def test_storage_model_survives_compression(tmp_path):
    """Compression must not disturb the storage model."""
    im = np.arange(4 * 6 * 3, dtype=np.uint16).reshape(4, 6, 3)
    path = tmp_path / "z.xisf"
    XISF.write(str(path), im, codec="zstd")
    assert read_image_attrs(path)["pixelStorage"] == "Planar"
    np.testing.assert_array_equal(XISF(str(path)).read_image(0), im)


# --------------------------------------------------------------------------
# Images of arbitrary dimensionality
#
# geometry is dim1:...:dimN:channel-count with N >= 1 (spec 11.5.1), so a 1-D
# image and a 4-D image are both legal. dim1 is the X-axis, which is the *last*
# numpy axis, so the spatial dimensions are written in reverse array order.
# --------------------------------------------------------------------------


# (numpy shape, expected geometry tuple, expected channels-last shape)
ND_CASES = [
    ((6,), (6, 1), (6, 1)),  # 1-D, single channel
    ((4, 6), (6, 4, 1), (4, 6, 1)),  # 2-D
    ((2, 2, 2), (2, 2, 2, 1), (2, 2, 2, 1)),  # 3-D, all spatial
    ((2, 3, 4, 5), (5, 4, 3, 2, 1), (2, 3, 4, 5, 1)),  # 4-D
    ((2, 2, 3, 2, 2), (2, 2, 3, 2, 2, 1), (2, 2, 3, 2, 2, 1)),  # 5-D
    ((1, 7), (7, 1, 1), (1, 7, 1)),  # degenerate dimension
]


@pytest.mark.parametrize("shape,geometry,expected", ND_CASES)
def test_nd_roundtrip(tmp_path, shape, geometry, expected):
    """Arrays of any dimensionality round-trip, with a conforming geometry."""
    im = np.arange(int(np.prod(shape)), dtype=np.uint16).reshape(shape)
    path = tmp_path / "nd.xisf"
    XISF.write(str(path), im)

    x = XISF(str(path))
    assert x.get_images_metadata()[0]["geometry"] == geometry
    assert read_image_attrs(path)["colorSpace"] == "Gray"

    back = x.read_image(0)
    assert back.shape == expected
    np.testing.assert_array_equal(back.reshape(shape), im)


@pytest.mark.parametrize("shape,geometry,expected", ND_CASES)
def test_nd_channels_first_output(tmp_path, shape, geometry, expected):
    """channels_first moves the channel axis to the front for any N."""
    im = np.arange(int(np.prod(shape)), dtype=np.uint16).reshape(shape)
    path = tmp_path / "ndcf.xisf"
    XISF.write(str(path), im)

    back = XISF(str(path)).read_image(0, data_format="channels_first")
    assert back.shape == (1, *shape)
    np.testing.assert_array_equal(back[0], im)


def test_1d_image_is_supported(tmp_path):
    """A 1-D image has N=1, so its geometry is dim1:channel-count.

    Regression test: a 1-D array raised IndexError: tuple index out of range,
    from reading im_data.shape[2] unconditionally.
    """
    im = np.arange(6, dtype=np.uint16)
    path = tmp_path / "n1.xisf"
    XISF.write(str(path), im)

    x = XISF(str(path))
    assert x.get_images_metadata()[0]["geometry"] == (6, 1)
    back = x.read_image(0)
    assert back.shape == (6, 1)
    np.testing.assert_array_equal(back[:, 0], im)


def test_4d_image_is_supported(tmp_path):
    """A 4-D array is four single-channel dimensions.

    Regression test: the geometry was built with '%d:%d:%d', so a 4-D array
    raised TypeError: not all arguments converted during string formatting.
    """
    im = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)
    path = tmp_path / "n4.xisf"
    XISF.write(str(path), im)

    x = XISF(str(path))
    assert x.get_images_metadata()[0]["geometry"] == (5, 4, 3, 2, 1)
    back = x.read_image(0)
    assert back.shape == (2, 3, 4, 5, 1)
    np.testing.assert_array_equal(back[..., 0], im)


def test_4d_image_with_channels(tmp_path):
    """A 4-D array can be channels first, so a 4-D image can be 3-channel."""
    cf = np.arange(3 * 2 * 3 * 4 * 5, dtype=np.uint16).reshape(3, 2, 3, 4, 5)
    path = tmp_path / "n4c.xisf"
    XISF.write(str(path), cf, pixel_storage="planar")

    x = XISF(str(path))
    # three spatial dimensions, then the channel count
    assert x.get_images_metadata()[0]["geometry"] == (5, 4, 3, 2, 3)
    assert read_image_attrs(path)["colorSpace"] == "RGB"

    back = x.read_image(0, data_format="channels_first")
    assert back.shape == (3, 2, 3, 4, 5)
    np.testing.assert_array_equal(back, cf)


def test_nd_image_with_channels_last(tmp_path):
    """A 4-D array read as normal storage has a trailing channel axis."""
    im = np.arange(2 * 3 * 4 * 1, dtype=np.uint16).reshape(2, 3, 4, 1)
    path = tmp_path / "n4n.xisf"
    XISF.write(str(path), im, pixel_storage="normal")

    x = XISF(str(path))
    assert x.get_images_metadata()[0]["geometry"] == (4, 3, 2, 1)
    back = x.read_image(0)
    assert back.shape == (2, 3, 4, 1)
    np.testing.assert_array_equal(back, im)


def test_nd_geometry_roundtrips_against_the_spec_order(tmp_path):
    """The data block order is independent of the geometry string order.

    dim1 is the X-axis and the first coordinate varies fastest in the planar
    model, so the samples are laid out in array order while the geometry lists
    the dimensions with the fastest-varying one first.
    """
    im = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)
    path = tmp_path / "order.xisf"
    XISF.write(str(path), im)

    x = XISF(str(path))
    assert x.get_images_metadata()[0]["geometry"] == (4, 3, 2, 1)
    # the raw block is the array in C order, single channel
    assert path.read_bytes()[-im.nbytes :] == im.tobytes()
    np.testing.assert_array_equal(x.read_image(0)[..., 0], im)


def test_geometry_with_one_item_is_rejected(tmp_path):
    """A geometry of a single item has no dimension, but N >= 1 (spec 11.5.1)."""
    with pytest.raises(XISFError, match="N >= 1"):
        XISF._parse_geometry("4")


def test_geometry_with_zero_item_is_rejected(tmp_path):
    """Every dimi item shall be greater than zero (spec 11.5.1)."""
    with pytest.raises(XISFError, match="greater than zero"):
        XISF._parse_geometry("4:0:1")


def test_pixel_storage_needs_two_dimensions(tmp_path):
    """Naming the channel axis explicitly requires something to name it from."""
    path = tmp_path / "bad.xisf"
    with pytest.raises(XISFError, match="at least two dimensions"):
        XISF.write(str(path), np.zeros(6, dtype=np.uint16), pixel_storage="normal")
    with pytest.raises(XISFError, match="at least two dimensions"):
        XISF.write(str(path), np.zeros(6, dtype=np.uint16), pixel_storage="planar")


# --------------------------------------------------------------------------
# The reader's output convention is unchanged
# --------------------------------------------------------------------------


def test_read_image_data_format_unchanged(tmp_path):
    """read_image still returns channels_last by default and can be flipped."""
    im = np.arange(4 * 6 * 3, dtype=np.uint16).reshape(4, 6, 3)
    path = tmp_path / "df.xisf"
    XISF.write(str(path), im)

    x = XISF(str(path))
    assert x.read_image(0).shape == (4, 6, 3)
    assert x.read_image(0, data_format="channels_first").shape == (3, 4, 6)
    np.testing.assert_array_equal(
        x.read_image(0, data_format="channels_first"), im.transpose(2, 0, 1)
    )


def test_roundtrip_preserves_dtypes(tmp_path):
    """Geometry and storage changes must not disturb sample format handling."""
    for dtype in (np.uint8, np.uint16, np.uint32, np.float32):
        im = (np.arange(4 * 6 * 3).reshape(4, 6, 3)).astype(dtype)
        path = tmp_path / f"d{dtype.__name__}.xisf"
        XISF.write(str(path), im)
        assert read_image_attrs(path)["sampleFormat"] == XISF._get_sampleFormat(
            np.dtype(dtype)
        )
        np.testing.assert_array_equal(XISF(str(path)).read_image(0), im)
