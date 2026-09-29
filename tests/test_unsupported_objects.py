"""Tests for the handling of unsupported objects required by section 7 of the
XISF 1.0 specification.

The specification requires that a decoder finding an object it does not support
shall treat that object as unavailable, report the situation to the user, and
keep the rest of the XISF unit accessible. The important part is the last clause:
one image the package cannot read must not make the other images of the same
file unreachable.
"""

import base64

import numpy as np
import pytest

from xisf import XISF, XISFError, XISFWarning

from .conftest import METADATA


def inline_image_element(im, sample_format="UInt8", **attrs):
    """An <Image> element with an inline Base64 data block."""
    w, h = im.shape[:2]
    c = im.shape[2] if im.ndim == 3 else 1
    extra = "".join(f' {k}="{v}"' for k, v in attrs.items())
    text = base64.b64encode(im.tobytes()).decode()
    return (
        f'<Image geometry="{w}:{h}:{c}" sampleFormat="{sample_format}"'
        f" colorSpace=\"Gray\" location=\"inline:base64\"{extra}>"
        f"{text}</Image>"
    )


def test_unsupported_sample_format_is_unavailable(make_file):
    """An unsupported sampleFormat shall not hide the images around it."""
    good = np.arange(4, dtype=np.uint8).reshape(2, 2)
    body = (
        METADATA
        + inline_image_element(good, sample_format="Int24")
        + inline_image_element(good)
    )
    path = make_file(body)

    with pytest.warns(XISFWarning, match="not supported"):
        x = XISF(str(path))

    # The unsupported image keeps its slot so that the indices of the images
    # after it do not shift
    assert len(x.get_images_metadata()) == 2
    assert "unavailable" in x.get_images_metadata()[0]
    assert "unavailable" not in x.get_images_metadata()[1]

    # The rest of the unit is accessible
    assert np.array_equal(x.read_image(1), good[:, :, None])


def test_reading_an_unavailable_image_raises(make_file):
    """Requesting an unavailable image shall say why, not fail obscurely."""
    good = np.arange(4, dtype=np.uint8).reshape(2, 2)
    body = METADATA + inline_image_element(good, sample_format="Int24")
    path = make_file(body)

    with pytest.warns(XISFWarning):
        x = XISF(str(path))

    with pytest.raises(XISFError, match="is unavailable: sampleFormat Int24"):
        x.read_image(0)


def test_unsupported_location_is_unavailable(make_file):
    """An unsupported data block location shall be handled the same way."""
    good = np.arange(4, dtype=np.uint8).reshape(2, 2)
    body = (
        METADATA
        + inline_image_element(good).replace(
            "inline:base64", "url:https://example.com/i.fits"
        )
        + inline_image_element(good)
    )
    path = make_file(body)

    with pytest.warns(XISFWarning, match="not supported"):
        x = XISF(str(path))

    assert len(x.get_images_metadata()) == 2
    assert np.array_equal(x.read_image(1), good[:, :, None])


def test_unsupported_codec_names_the_owner(make_file):
    """An unsupported codec shall name the object it belongs to.

    The failure surfaces at read time, so the rest of the unit is reachable
    either way; naming the owner is what makes the report actionable.
    """
    good = np.arange(4, dtype=np.uint8).reshape(2, 2)
    body = (
        METADATA
        + inline_image_element(good, id="broken", compression="brotli:4")
        + inline_image_element(good, id="fine")
    )
    path = make_file(body)

    x = XISF(str(path))

    with pytest.raises(NotImplementedError, match="Image broken is not supported"):
        x.read_image(0)

    assert np.array_equal(x.read_image(1), good[:, :, None])


def test_unsupported_encoding_names_the_owner(make_file):
    """An unsupported encoding shall name the object it belongs to."""
    good = np.arange(4, dtype=np.uint8).reshape(2, 2)
    body = (
        METADATA
        + inline_image_element(good, id="broken").replace(
            "inline:base64", "inline:rot13"
        )
        + inline_image_element(good, id="fine")
    )
    path = make_file(body)

    x = XISF(str(path))

    with pytest.raises(NotImplementedError, match="Image broken is not supported"):
        x.read_image(0)

    assert np.array_equal(x.read_image(1), good[:, :, None])


def test_all_images_unsupported_still_opens(make_file):
    """A unit whose every image is unsupported shall still be openable.

    Nothing can be read out of it, but the unit itself is not corrupt and the
    user can inspect why each image was skipped.
    """
    im = np.arange(4, dtype=np.uint8).reshape(2, 2)
    body = (
        METADATA
        + inline_image_element(im, sample_format="Int24")
        + inline_image_element(im, sample_format="Float24")
    )
    path = make_file(body)

    with pytest.warns(XISFWarning):
        x = XISF(str(path))

    assert len(x.get_images_metadata()) == 2
    for n in range(2):
        with pytest.raises(XISFError, match="is unavailable"):
            x.read_image(n)
