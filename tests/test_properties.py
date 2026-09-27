"""Tests for XISF property type dispatch in _process_property.

Sections 11.1.4 to 11.1.9 of the XISF 1.0 specification constrain which
attributes a Property element of each type shall or shall not carry:

  * scalar and Complex  "shall have a value attribute"      (11.1.4, 11.1.5)
  * String              "shall not have a value attribute"   (11.1.6)
  * Vector              "Shall not have a value attribute"   (11.1.8)
  * Matrix              "Shall not have a value attribute"   (11.1.9)

and section 7 requires that "XML elements, XML attributes and properties that
a decoder does not recognize shall be ignored", so an attribute that is not
valid for a type must not change how that property is decoded.
"""

import warnings

import numpy as np
import pytest

from xisf import XISF, XISFWarning

from .conftest import (
    METADATA,
    image_element,
    matrix_property_element,
    sample_image,
    vector_property_element,
)


def body_with_image(inner):
    """Wrap property elements inside a real image so the unit is readable."""
    return METADATA + image_element(sample_image())[: -len("</Image>")] + inner + "</Image>"


def image_props(make_file, inner, name="props.xisf"):
    """Write a file with one real image carrying the given property elements."""
    return make_file(body_with_image(inner), name=name)


def read_props(path):
    return XISF(str(path)).get_images_metadata()[0]["XISFProperties"]


# --------------------------------------------------------------------------
# Well-formed properties: type dispatch must be correct and silent
# --------------------------------------------------------------------------


def test_vector_without_value(make_file):
    values = np.array([1.0, 2.0, 3.0, 4.0])
    path = image_props(make_file, vector_property_element(values))

    with _no_warnings():
        props = read_props(path)

    prop = props["F64Vector:Test"]
    assert prop["dtype"] == np.dtype("float64")
    assert prop["length"] == 4
    assert isinstance(prop["length"], int)
    np.testing.assert_array_equal(prop["value"], values)


def test_matrix_without_value(make_file):
    values = np.arange(6, dtype=np.float64).reshape(2, 3)
    path = image_props(make_file, matrix_property_element(values))

    with _no_warnings():
        props = read_props(path)

    prop = props["F64Matrix:Test"]
    assert prop["dtype"] == np.dtype("float64")
    assert prop["rows"] == 2 and prop["columns"] == 3
    assert isinstance(prop["rows"], int)
    assert prop["value"].shape == (2, 3)
    np.testing.assert_array_equal(prop["value"], values)


# --------------------------------------------------------------------------
# The regression: a forbidden value attribute must not hijack Vector/Matrix
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra_value", ["1.5", "2.5", "not-a-number", "SOME GARBAGE", "1.5+2.5j"]
)
def test_vector_with_forbidden_value_still_decodes(make_file, extra_value):
    """A value attribute on a Vector is ignored; the vector still decodes.

    Regression test: the scalar branch was tested before the Vector branch, so
    any value attribute replaced the vector with ast.literal_eval(value), and a
    non-literal string raised ValueError at open time.
    """
    values = np.array([1.0, 2.0, 3.0, 4.0])
    path = image_props(
        make_file,
        vector_property_element(values, extra_attrs=f' value="{extra_value}"'),
    )

    with pytest.warns(XISFWarning, match="forbidden value attribute"):
        props = read_props(path)

    np.testing.assert_array_equal(props["F64Vector:Test"]["value"], values)


@pytest.mark.parametrize("extra_value", ["1.5", "not-a-number", "SOME GARBAGE"])
def test_matrix_with_forbidden_value_still_decodes(make_file, extra_value):
    """A value attribute on a Matrix is ignored; the matrix still decodes."""
    values = np.arange(6, dtype=np.float64).reshape(2, 3)
    path = image_props(
        make_file,
        matrix_property_element(values, extra_attrs=f' value="{extra_value}"'),
    )

    with pytest.warns(XISFWarning, match="forbidden value attribute"):
        props = read_props(path)

    prop = props["F64Matrix:Test"]
    assert prop["value"].shape == (2, 3)
    np.testing.assert_array_equal(prop["value"], values)
    # The scalar branch used to leave these as strings
    assert isinstance(prop["rows"], int)
    assert isinstance(prop["columns"], int)


def test_forbidden_value_does_not_crash_open(make_file):
    """A non-literal value must not prevent the file from opening at all.

    Regression test: ast.literal_eval("SOME GARBAGE") raised ValueError from
    XISF.__init__, making the whole unit unreadable.
    """
    values = np.arange(6, dtype=np.float64).reshape(2, 3)
    path = make_file(
        body_with_image(
            matrix_property_element(values, extra_attrs=' value="SOME GARBAGE"')
        )
    )

    with pytest.warns(XISFWarning, match="forbidden value attribute"):
        x = XISF(str(path))
    # The rest of the unit stays accessible, as section 7 requires
    assert x.read_image(0).shape == sample_image().shape
    assert x.get_file_metadata()["XISF:Keep"]["value"] == 1


def test_forbidden_value_in_file_metadata(make_file):
    """The same rule applies to properties in the file-level Metadata element."""
    values = np.array([7.0, 8.0])
    body = (
        '<Metadata><Property id="X:Keep" type="Int" value="1"/>'
        + vector_property_element(values, extra_attrs=' value="1.5"')
        + "</Metadata>"
        + '<Image geometry="2:2:1" sampleFormat="UInt16" colorSpace="Gray"'
        ' location="attachment:4096:0"/>'
    )
    path = make_file(body)

    with pytest.warns(XISFWarning, match="forbidden value attribute"):
        fm = XISF(str(path)).get_file_metadata()

    np.testing.assert_array_equal(fm["F64Vector:Test"]["value"], values)


# --------------------------------------------------------------------------
# Scalars must keep working: the reorder must not steal the value branch
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ptype,text,expected",
    [
        ("Int", "42", 42),
        ("Int16", "-7", -7),
        ("UInt32", "4000000000", 4000000000),
        ("Float32", "3.5", 3.5),
        ("Float64", "-1.25e-3", -1.25e-3),
        ("C64", "1.5+2.5j", 1.5 + 2.5j),
        ("C32", "0.5-1.5j", 0.5 - 1.5j),
    ],
)
def test_scalar_properties_unaffected(make_file, ptype, text, expected):
    """Scalar and Complex properties still take the value branch."""
    body = body_with_image(f'<Property id="S:Test" type="{ptype}" value="{text}"/>')
    path = make_file(body)

    with _no_warnings():
        props = read_props(path)

    got = props["S:Test"]["value"]
    assert got == expected
    if isinstance(expected, complex):
        assert isinstance(got, complex)


@pytest.mark.parametrize("text,expected", [("true", True), ("false", False)])
def test_boolean_property_unaffected(make_file, text, expected):
    """Boolean properties are handled by their own branch, not the value one."""
    body = body_with_image(f'<Property id="B:Test" type="Boolean" value="{text}"/>')
    path = make_file(body)
    with _no_warnings():
        props = read_props(path)
    assert props["B:Test"]["value"] is expected


def test_timepoint_property_unaffected(make_file):
    body = body_with_image(
        '<Property id="T:Test" type="TimePoint" value="2026-09-27T20:57:35"/>'
    )
    path = make_file(body)
    with _no_warnings():
        props = read_props(path)
    assert props["T:Test"]["value"] == "2026-09-27T20:57:35"


# --------------------------------------------------------------------------
# Genuinely unsupported types must still be reported as unsupported
# --------------------------------------------------------------------------


def test_unsupported_type_still_reported(make_file):
    """A Table property is genuinely unsupported and must still be reported."""
    body = body_with_image('<Property id="T:Table" type="Table" rows="1"/>')
    path = make_file(body)

    with pytest.warns(XISFWarning, match="Unsupported Property type Table"):
        props = read_props(path)

    assert "T:Table" not in props  # dropped for image properties


def test_ordering_is_stable_when_property_id_collides(make_file):
    """Sanity check that a single file with many property types parses cleanly."""
    values = np.arange(4, dtype=np.float64)
    inner = (
        vector_property_element(values)
        + matrix_property_element(values.reshape(2, 2))
        + '<Property id="S:Int" type="Int" value="9"/>'
        + '<Property id="B:Flag" type="Boolean" value="true"/>'
    )
    path = image_props(make_file, inner)

    with _no_warnings():
        props = read_props(path)

    assert set(props) == {
        "F64Vector:Test",
        "F64Matrix:Test",
        "S:Int",
        "B:Flag",
    }
    assert props["S:Int"]["value"] == 9
    assert props["B:Flag"]["value"] is True


# --------------------------------------------------------------------------
# Round-trip through the writer
# --------------------------------------------------------------------------


def test_vector_matrix_roundtrip_through_writer(tmp_path):
    """Matrices and vectors written by XISF.write() must read back intact."""
    from xisf import XISF as Writer

    im = np.arange(4 * 6 * 3, dtype=np.uint16).reshape(4, 6, 3)
    props = {
        "V": {"id": "V", "type": "F64Vector", "value": np.array([1.5, 2.5, 3.5])},
        "M": {
            "id": "M",
            "type": "F64Matrix",
            "value": np.arange(6, dtype=np.float64).reshape(2, 3),
        },
    }
    path = tmp_path / "rt.xisf"
    Writer.write(str(path), im, image_metadata={"XISFProperties": props})

    with _no_warnings():
        back = Writer(str(path)).get_images_metadata()[0]["XISFProperties"]

    np.testing.assert_array_equal(back["V"]["value"], props["V"]["value"])
    np.testing.assert_array_equal(back["M"]["value"], props["M"]["value"])


class _no_warnings:
    """Assert that reading produced no warnings at all."""

    def __enter__(self):
        self._mgr = warnings.catch_warnings(record=True)
        self._log = self._mgr.__enter__()
        warnings.simplefilter("always")
        return self

    def __exit__(self, exc_type, exc, tb):
        self._mgr.__exit__(exc_type, exc, tb)
        if exc_type is None:
            assert not self._log, (
                f"unexpected warnings: {[str(w.message) for w in self._log]}"
            )
        return False
