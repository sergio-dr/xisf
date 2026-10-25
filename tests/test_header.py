"""XISF 1.0 spec, 9.5: structure of a valid XISF header.

Two mandatory parts of the XISF header were historically missing from the
hand-built headers written by conftest.write_monolithic(): the XML
declaration and the version attribute of the root element. Because the
library does not enforce either on read, the whole suite passed while every
file the helper produced was invalid. XISF.write() had the same class of
problem: it let ET.tostring() write the declaration, which does not match
the one this specification requires. These tests pin down the requirements
and keep the helper in sync with the attributes XISF.write() emits.
"""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from xisf import XISF

from .conftest import (
    METADATA,
    ROOT_ATTRS,
    XML_DECL,
    attached_image,
    image_element,
    sample_image,
    write_monolithic,
)

XSI = "{http://www.w3.org/2001/XMLSchema-instance}"
IMAGE = f"{{{XISF._xml_ns['xisf']}}}Image"


def header_of(path):
    """Return the raw XML header of a monolithic XISF file."""
    with open(path, "rb") as f:
        raw = f.read()
    headerlength = int.from_bytes(raw[8:12], "little")
    return raw[16 : 16 + headerlength]


def written_header(tmp_path):
    path = tmp_path / "hand_built.xisf"
    write_monolithic(str(path), METADATA)
    return header_of(path)


def test_xml_declaration_is_first_byte(tmp_path):
    """9.5: a valid XISF header must begin with the XML declaration."""
    assert written_header(tmp_path).startswith(XML_DECL.encode())


def test_root_element_has_version(tmp_path):
    """9.5: the root element must have a version attribute equal to "1.0"."""
    root = ET.fromstring(written_header(tmp_path))
    assert root.get("version") == "1.0"


def test_root_element_has_no_character_data(tmp_path):
    """9.5: the root element must not have character data contents."""
    root = ET.fromstring(written_header(tmp_path))
    assert not (root.text or "").strip()


def test_root_element_has_namespace_and_schema(tmp_path):
    """9.5: the root element should declare the XISF and xsi namespaces."""
    root = ET.fromstring(written_header(tmp_path))
    assert root.tag == f"{{{XISF._xml_ns['xisf']}}}xisf"
    assert root.get(f"{XSI}schemaLocation") is not None


def test_root_attrs_mirror_the_writer():
    """The hand-built root start-tag must match XISF.write()'s attributes."""
    expected = " ".join(f'{k}="{v}"' for k, v in XISF._xisf_attrs.items())
    assert ROOT_ATTRS == expected


def test_written_file_satisfies_root_requirements(tmp_path):
    """XISF.write() output, checked against the same 9.5 requirements."""
    path = tmp_path / "written.xisf"
    XISF.write(str(path), sample_image(), checksum=False)
    root = ET.fromstring(header_of(path))
    assert root.tag == f"{{{XISF._xml_ns['xisf']}}}xisf"
    assert root.get("version") == "1.0"
    assert root.get(f"{XSI}schemaLocation") is not None
    assert not (root.text or "").strip()


def test_written_file_starts_with_xml_declaration(tmp_path):
    """9.5: XISF.write() must begin the header with the XML declaration."""
    path = tmp_path / "decl.xisf"
    XISF.write(str(path), sample_image(), checksum=False)
    assert header_of(path).startswith(XML_DECL.encode("utf-8"))


def test_bounds_required_for_float_images(tmp_path):
    """11.5.2: bounds is required for Float32 and Float64 images."""
    for dtype in (np.float32, np.float64):
        path = tmp_path / f"{dtype.__name__}.xisf"
        write_monolithic(str(path), METADATA + image_element(sample_image(dtype=dtype)))
        image = ET.fromstring(header_of(path)).find(IMAGE)
        assert image.get("bounds") == "0:1"


def test_bounds_honours_an_explicit_range(tmp_path):
    """The representable range can be overridden, not just defaulted."""
    path = tmp_path / "bounded.xisf"
    write_monolithic(
        str(path),
        METADATA + image_element(sample_image(dtype=np.float32), bounds="-1:1"),
    )
    image = ET.fromstring(header_of(path)).find(IMAGE)
    assert image.get("bounds") == "-1:1"


def test_integer_images_omit_bounds(tmp_path):
    """11.5.2: integer images default to their full range, so bounds stays out."""
    path = tmp_path / "integer.xisf"
    write_monolithic(str(path), METADATA + image_element(sample_image()))
    image = ET.fromstring(header_of(path)).find(IMAGE)
    assert image.get("bounds") is None


def test_attached_float_image_declares_bounds(tmp_path):
    """The attached-block builder must satisfy 11.5.2 as well."""
    path = attached_image(tmp_path / "attached.xisf", sample_image(dtype=np.float32))
    image = ET.fromstring(header_of(path)).find(IMAGE)
    assert image.get("bounds") == "0:1"


def test_metadata_has_the_mandatory_properties(tmp_path):
    """11.4.1: Metadata must serialize CreationTime and CreatorApplication."""
    path = tmp_path / "metadata.xisf"
    write_monolithic(str(path), METADATA + image_element(sample_image()))
    meta = XISF(str(path)).get_file_metadata()
    assert meta["XISF:CreationTime"]["type"] == "TimePoint"
    assert meta["XISF:CreatorApplication"]["type"] == "String"
    assert meta["XISF:CreatorApplication"]["value"] == "xisf test suite"
