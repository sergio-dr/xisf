"""Tests for the vector and matrix property types of spec 8.4.4.5 and 8.4.4.6.

Both sections declare a table of type names, some of which have an alternate
name. A property type cannot be classified from its spelling, because an
alternate name need not contain the suffix that identifies the property: the
alternate name ByteArray denotes a vector, and contains the name of a scalar
element type. These tests cover every type name of both tables, in both the
reading and the writing direction.
"""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from xisf import XISF, XISFError, XISFWarning

from .conftest import (
    METADATA,
    COMPRESSORS,
    attached_image,
    matrix_property_element,
    sample_image,
    vector_property_element,
    write_monolithic,
)

# Element type of every type name of spec 8.4.4.5 Table 7 (vectors) and 8.4.4.6
# Table 8 (matrices), which declare the same element types. The alternate names
# are those declared in the "Alternate name" column of the tables.
VECTOR_DTYPES = {
    "I8Vector": "int8",
    "UI8Vector": "uint8",
    "ByteArray": "uint8",
    "I16Vector": "int16",
    "UI16Vector": "uint16",
    "I32Vector": "int32",
    "IVector": "int32",
    "UI32Vector": "uint32",
    "UIVector": "uint32",
    "I64Vector": "int64",
    "UI64Vector": "uint64",
    "F32Vector": "float32",
    "F64Vector": "float64",
    "Vector": "float64",
}
MATRIX_DTYPES = {
    "I8Matrix": "int8",
    "UI8Matrix": "uint8",
    "ByteMatrix": "uint8",
    "I16Matrix": "int16",
    "UI16Matrix": "uint16",
    "I32Matrix": "int32",
    "IMatrix": "int32",
    "UI32Matrix": "uint32",
    "UIMatrix": "uint32",
    "I64Matrix": "int64",
    "UI64Matrix": "uint64",
    "F32Matrix": "float32",
    "F64Matrix": "float64",
    "Matrix": "float64",
}
# The alternate name of each type, as declared by the tables
ALTERNATE_OF = {
    "UI8Vector": "ByteArray",
    "I32Vector": "IVector",
    "UI32Vector": "UIVector",
    "F64Vector": "Vector",
    "UI8Matrix": "ByteMatrix",
    "I32Matrix": "IMatrix",
    "UI32Matrix": "UIMatrix",
    "F64Matrix": "Matrix",
}
# The 128-bit types of both tables, whose support is optional and is not
# implemented. numpy has no 128-bit dtype, so they cannot round-trip.
UNIMPLEMENTED = ["I128Vector", "UI128Vector", "F128Vector", "C128Vector"]


def write_property(tmp_path, ptype, value, name="p.xisf"):
    XISF.write(
        str(tmp_path / name),
        sample_image(),
        "xisf.test",
        image_metadata={
            "XISFProperties": {"P": {"id": "P", "type": ptype, "value": value}}
        },
    )


def read_property(path, prop_id="P", as_file=True):
    """Read a property back, from the file metadata or an image's properties."""
    x = XISF(str(path))
    if as_file:
        return x.get_file_metadata()[prop_id]
    return x.get_images_metadata()[0]["XISFProperties"][prop_id]


def read_written_property(path, prop_id="P"):
    """Read back a property written by write_property, which is an image one."""
    return read_property(path, prop_id, as_file=False)


def serialized_property_element(path, prop_id="P"):
    """The serialized <Property> element of a written file.

    A vector or a matrix is serialized as a data block, so the element carries a
    location and a length, rows and columns, and no value attribute. Reading the
    property back cannot show that, since the parsed dict holds the value under
    the name 'value' in either serialization.
    """
    raw = path.read_bytes()
    header = raw[16 : 16 + int.from_bytes(raw[8:12], "little")].decode("utf-8")
    for element in ET.fromstring(header).iter():
        # the elements are namespaced, so the local name is compared
        if element.tag.endswith("Property") and element.get("id") == prop_id:
            return element
    raise AssertionError(f"no Property {prop_id!r} in the written header")


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ptype,dtype", list(VECTOR_DTYPES.items()))
def test_every_vector_type_name_is_read(make_file, ptype, dtype):
    """Every type name of Table 7 shall be read as a vector."""
    arr = np.array([1, 2, 3], dtype=dtype)
    path = make_file(
        METADATA[:-len("</Metadata>")]
        + vector_property_element(arr, ptype=ptype)
        + "</Metadata>",
        name=f"{ptype}.xisf",
    )
    prop = read_property(path, prop_id=f"{ptype}:Test")
    assert prop["value"].dtype == np.dtype(dtype)
    assert np.array_equal(prop["value"], arr)


@pytest.mark.parametrize("ptype,dtype", list(MATRIX_DTYPES.items()))
def test_every_matrix_type_name_is_read(make_file, ptype, dtype):
    """Every type name of Table 8 shall be read as a matrix."""
    arr = np.array([[1, 2], [3, 4]], dtype=dtype)
    path = make_file(
        METADATA[:-len("</Metadata>")]
        + matrix_property_element(arr, ptype=ptype)
        + "</Metadata>",
        name=f"{ptype}.xisf",
    )
    prop = read_property(path, prop_id=f"{ptype}:Test")
    assert prop["value"].dtype == np.dtype(dtype)
    assert prop["value"].shape == (2, 2)
    assert np.array_equal(prop["value"], arr)


def test_alternate_name_is_read_as_its_canonical_type(make_file):
    """An alternate name shall read back with the dtype of its canonical name."""
    arr = np.array([1, 2, 3], dtype="uint8")
    path = make_file(
        METADATA[:-len("</Metadata>")]
        + vector_property_element(arr, ptype="ByteArray")
        + "</Metadata>",
        name="bytearray.xisf",
    )
    prop = read_property(path, prop_id="ByteArray:Test")
    # ByteArray is the alternate name of UI8Vector (spec 8.4.4.5 Table 7)
    assert prop["value"].dtype == np.dtype("uint8")
    assert np.array_equal(prop["value"], arr)


def test_forbidden_value_attribute_on_a_matrix_is_reported(make_file):
    """A Matrix property shall not have a value attribute (spec 11.1.9).

    The check must also catch the alternate name ByteMatrix, whose name
    contains a scalar element type.
    """
    arr = np.array([[1, 2], [3, 4]], dtype="uint8")
    element = matrix_property_element(arr, ptype="ByteMatrix").replace(
        'type="ByteMatrix"', 'type="ByteMatrix" value="[1, 2]"'
    )
    path = make_file(
        METADATA[:-len("</Metadata>")] + element + "</Metadata>",
        name="forbidden.xisf",
    )
    with pytest.warns(XISFWarning, match="forbidden value attribute"):
        x = XISF(str(path))
    assert "ByteMatrix:Test" in x.get_file_metadata()


@pytest.mark.parametrize("ptype", UNIMPLEMENTED)
def test_128_bit_types_are_reported_as_unimplemented_data_types(make_file, ptype):
    """A 128-bit type shall be recognized, then reported as unimplemented.

    It is a vector or a matrix of Tables 7 and 8, so it is not an unknown
    property type, but this package has no dtype for it.
    """
    path = make_file(
        METADATA[:-len("</Metadata>")]
        + vector_property_element(b"\0" * 16, ptype=ptype)
        + "</Metadata>",
        name=f"{ptype}.xisf",
    )
    with pytest.raises(NotImplementedError, match=f"data type {ptype} not implemented"):
        XISF(str(path))


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ptype,dtype", list(VECTOR_DTYPES.items()))
def test_every_vector_type_name_is_written(tmp_path, ptype, dtype):
    """Every type name of Table 7 shall be written as a data block."""
    arr = np.array([1, 2, 3], dtype=dtype)
    write_property(tmp_path, ptype, arr)
    prop = read_written_property(tmp_path / "p.xisf")
    assert prop["value"].dtype == np.dtype(dtype)
    assert np.array_equal(prop["value"], arr)
    # A vector is serialized as a data block, never as a value attribute
    element = serialized_property_element(tmp_path / "p.xisf")
    assert element.get("length") == "3"
    assert element.get("value") is None


@pytest.mark.parametrize("ptype,dtype", list(MATRIX_DTYPES.items()))
def test_every_matrix_type_name_is_written(tmp_path, ptype, dtype):
    """Every type name of Table 8 shall be written as a data block."""
    arr = np.array([[1, 2], [3, 4]], dtype=dtype)
    write_property(tmp_path, ptype, arr)
    prop = read_written_property(tmp_path / "p.xisf")
    assert prop["value"].dtype == np.dtype(dtype)
    assert np.array_equal(prop["value"], arr)
    element = serialized_property_element(tmp_path / "p.xisf")
    assert (element.get("rows"), element.get("columns")) == ("2", "2")
    assert element.get("value") is None


@pytest.mark.parametrize("ptype", ["ByteArray", "ByteMatrix"])
def test_alternate_name_is_written_as_its_canonical_name(tmp_path, ptype):
    """An alternate name shall be written under its canonical name.

    ByteArray and ByteMatrix contain the name of a scalar element type, so they
    are the two names that are misclassified if a vector or a matrix is
    recognized by its spelling.
    """
    is_matrix = ptype.endswith("Matrix")
    arr = (
        np.array([[1, 2], [3, 4]], dtype="uint8")
        if is_matrix
        else np.array([1, 2, 3], dtype="uint8")
    )
    write_property(tmp_path, ptype, arr)
    canonical = "UI8Matrix" if is_matrix else "UI8Vector"
    prop = read_written_property(tmp_path / "p.xisf")
    assert prop["type"] == canonical
    assert np.array_equal(prop["value"], arr)


@pytest.mark.parametrize("ptype,alternate", sorted(ALTERNATE_OF.items()))
def test_canonical_name_round_trips(tmp_path, ptype, alternate):
    """A canonical type name shall be read and written unchanged.

    The alternate name of a type denotes the same property, so both spellings
    have to give the same data, and the canonical one is not rewritten.
    """
    is_matrix = ptype.endswith("Matrix")
    dtype = (MATRIX_DTYPES if is_matrix else VECTOR_DTYPES)[ptype]
    arr = (
        np.array([[1, 2], [3, 4]], dtype=dtype)
        if is_matrix
        else np.array([1, 2, 3], dtype=dtype)
    )
    write_property(tmp_path, ptype, arr)
    prop = read_written_property(tmp_path / "p.xisf")
    assert prop["type"] == ptype
    assert prop["value"].dtype == np.dtype(dtype)
    assert np.array_equal(prop["value"], arr)

    # The alternate name gives exactly the same result
    write_property(tmp_path, alternate, arr, name="alt.xisf")
    alt_prop = read_written_property(tmp_path / "alt.xisf")
    assert alt_prop["type"] == ptype
    assert np.array_equal(alt_prop["value"], arr)


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ptype,is_vector_or_matrix",
    [
        ("I8Vector", True),
        ("UI8Matrix", True),
        ("ByteArray", True),
        ("ByteMatrix", True),
        ("IVector", True),
        ("UIMatrix", True),
        ("Vector", True),
        ("Matrix", True),
        ("I128Vector", True),
        ("C64Matrix", True),
        ("Int8", False),
        ("Float64", False),
        ("String", False),
        ("Boolean", False),
        ("TimePoint", False),
        # A name that only ends with the suffix is not a type of either table
        ("Vector64", False),
        ("MyMatrix", False),
        ("NotAType", False),
    ],
)
def test_vector_and_matrix_types_are_classified(ptype, is_vector_or_matrix):
    """A type is a vector or a matrix only if Tables 7 and 8 declare it."""
    assert XISF._is_vector_or_matrix(ptype) is is_vector_or_matrix


def test_vector_matrix_dtypes_cover_the_required_element_types():
    """Tables 7 and 8 require 8, 16, 32 and 64-bit element types.

    Support for the 128-bit types is optional, so they are not required here.
    """
    for prefix in ("I8", "UI8", "I16", "UI16", "I32", "UI32", "I64", "UI64",
                   "F32", "F64"):
        assert XISF._parse_vector_matrix_dtype(prefix + "Vector") is not None
        assert XISF._parse_vector_matrix_dtype(prefix + "Matrix") is not None


# --------------------------------------------------------------------------
# Compressed data blocks
# --------------------------------------------------------------------------

CODECS = [None, "zlib", "lz4", "lz4hc", "zstd"]
LOCATIONS = ["inline", "embedded"]
# The item size of a byte-shuffled F64 block, which is the element size
ITEM_SIZE = 8
VALUES = np.array([1.5, -2.5, 3.25, 4.0, 5.5, 6.75, 7.0, 8.25], dtype="float64")


def file_with_property(path, element, attached=b""):
    """A file whose Metadata holds the one property element."""
    write_monolithic(
        path, METADATA[: -len("</Metadata>")] + element + "</Metadata>", attached
    )
    return str(path)


def file_with_attached_property(path, ptype, shape_attrs, arr, codec, checksum=None):
    """A file whose Metadata holds a property in an attached compressed block.

    The offset of an attached block is only known once the header is sized, so
    it is measured with an empty body first. Attached blocks are aligned to
    4096 bytes, so the offset does not move once the real header is written, as
    long as the header stays under one block.
    """
    block = COMPRESSORS[codec](arr.tobytes())
    offset = write_monolithic(path, "", attached=block)
    attrs = "".join(f' {k}="{v}"' for k, v in shape_attrs.items())
    digest = f' checksum="sha-1:{checksum}"' if checksum else ""
    element = (
        f'<Property id="P" type="{ptype}"{attrs} compression="{codec}:{arr.nbytes}"'
        f'{digest} location="attachment:{offset}:{len(block)}" />'
    )
    return file_with_property(path, element, attached=block)


@pytest.mark.parametrize("codec", CODECS)
@pytest.mark.parametrize("location", LOCATIONS)
@pytest.mark.parametrize("shuffled", [False, True])
def test_compressed_vector_is_read(make_file, codec, location, shuffled):
    """A Vector property shall be read from a compressed data block."""
    element = vector_property_element(
        VALUES,
        ptype="F64Vector",
        location=location,
        codec=codec,
        item_size=ITEM_SIZE if shuffled else None,
    )
    prop = XISF(file_with_property(make_file(element), element)).get_file_metadata()[
        "F64Vector:Test"
    ]
    assert prop["value"].dtype == np.dtype("float64")
    assert np.array_equal(prop["value"], VALUES)


@pytest.mark.parametrize("codec", CODECS)
@pytest.mark.parametrize("location", LOCATIONS)
@pytest.mark.parametrize("shuffled", [False, True])
def test_compressed_matrix_is_read(make_file, codec, location, shuffled):
    """A Matrix property shall be read from a compressed data block."""
    arr = VALUES.reshape(4, 2)
    element = matrix_property_element(
        arr,
        ptype="F64Matrix",
        location=location,
        codec=codec,
        item_size=ITEM_SIZE if shuffled else None,
    )
    prop = XISF(file_with_property(make_file(element), element)).get_file_metadata()[
        "F64Matrix:Test"
    ]
    assert prop["value"].dtype == np.dtype("float64")
    assert prop["value"].shape == (4, 2)
    assert np.array_equal(prop["value"], arr)


@pytest.mark.parametrize("codec", [c for c in CODECS if c])
def test_attached_compressed_vector_is_read(tmp_path, codec):
    """A Vector property shall be read from an attached compressed block."""
    path = file_with_attached_property(
        tmp_path / f"a-{codec}.xisf",
        "F64Vector",
        {"length": VALUES.size},
        VALUES,
        codec,
    )
    prop = XISF(path).get_file_metadata()["P"]
    assert np.array_equal(prop["value"], VALUES)


@pytest.mark.parametrize("codec", [c for c in CODECS if c])
def test_attached_compressed_matrix_is_read(tmp_path, codec):
    """A Matrix property shall be read from an attached compressed block."""
    arr = VALUES.reshape(4, 2)
    path = file_with_attached_property(
        tmp_path / f"a-{codec}.xisf",
        "F64Matrix",
        {"rows": 4, "columns": 2},
        arr,
        codec,
    )
    prop = XISF(path).get_file_metadata()["P"]
    assert prop["value"].shape == (4, 2)
    assert np.array_equal(prop["value"], arr)
