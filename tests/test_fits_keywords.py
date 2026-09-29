"""Tests for FITSKeyword core elements and the FITS standard's keyword rules.

Section 11.6.1 of the XISF 1.0 specification declares three mandatory
attributes on a FITSKeyword element:

  * name    the FITS header keyword name
  * value   the keyword value, an empty string for HISTORY and COMMENT, which
            have no value of their own
  * comment the keyword comment

and quotes the FITS standard for the name, which shall be a left justified,
space-filled ASCII string with no embedded spaces, in which only the digits
0-9, the upper case Latin letters A-Z, the underscore and the hyphen are
permitted. In FITSKeyword elements names must not be padded with spaces.

A conforming encoder shall generate only valid units (section 7), so writing an
invalid name is an error. A decoder that finds an object it does not support
shall keep the rest of the unit accessible (section 7), so reading a malformed
keyword is reported as a warning and recovered from.
"""

import warnings

import numpy as np
import pytest

from xisf import XISF, XISFError, XISFWarning

from .conftest import METADATA, image_element, sample_image, write_monolithic


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
                f"unexpected warnings: {[str(m.message) for m in self._log]}"
            )
        return False


def keyword_file(tmp_path, keyword_elements):
    """Write a file whose Image carries the given FITSKeyword child elements."""
    image = image_element(sample_image())[: -len("</Image>")]
    body = METADATA + image + "".join(keyword_elements) + "</Image>"
    path = tmp_path / "kw.xisf"
    write_monolithic(str(path), body)
    return str(path)


def read_keywords(path):
    with _no_warnings():
        return XISF(path).get_images_metadata()[0]["FITSKeywords"]


# --------------------------------------------------------------------------
# Valid keywords are written and read unchanged
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["EXPTIME", "NAXIS1", "XBINNING", "A_B", "A-B", "1234", "OBSERVER"]
)
def test_valid_keyword_name_is_accepted(tmp_path, name):
    """The FITS standard permits digits, A-Z, underscore and hyphen, up to 8."""
    path = tmp_path / "kw.xisf"
    XISF.write(
        str(path),
        sample_image(),
        "test",
        {"FITSKeywords": {name: [{"value": "1", "comment": "c"}]}},
    )
    assert read_keywords(str(path))[name] == [{"value": "1", "comment": "c"}]


# --------------------------------------------------------------------------
# Invalid keyword names: an error when writing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["exptime", "TOO_LONG_NAME", "NAXIS 1", "EXPTIME ", "", "NAXIS:1", "ÉXPT"]
)
def test_invalid_keyword_name_is_refused_when_writing(tmp_path, name):
    """An encoder shall generate only units that satisfy the spec (section 7)."""
    with pytest.raises(XISFError, match="11.6.1"):
        XISF.write(
            str(tmp_path / "kw.xisf"),
            sample_image(),
            "test",
            {"FITSKeywords": {name: [{"value": "1", "comment": "c"}]}},
        )


# --------------------------------------------------------------------------
# Invalid keyword names: a warning and a sanitized name when reading
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected", [("exptime", "EXPTIME"), ("exp time", "EXP_TIME"), ("x" * 20, "X" * 8)]
)
def test_invalid_keyword_name_warns_and_is_sanitized(tmp_path, name, expected):
    """A malformed name is repaired and reported, not raised on."""
    path = keyword_file(
        tmp_path, [f'<FITSKeyword name="{name}" value="1" comment="c"/>']
    )
    with pytest.warns(XISFWarning, match="11.6.1"):
        keywords = XISF(path).get_images_metadata()[0]["FITSKeywords"]
    assert list(keywords) == [expected]
    assert keywords[expected] == [{"value": "1", "comment": "c"}]


def test_keyword_name_of_only_invalid_characters(tmp_path):
    """A name with no usable character left is replaced, and reported.

    Every character is outside the FITS charset, so the name cannot be kept as
    it is. The keyword is still readable under a placeholder name rather than
    being dropped, since a decoder must keep the unit accessible.
    """
    path = keyword_file(tmp_path, ['<FITSKeyword name="!!" value="1" comment="c"/>'])
    with pytest.warns(XISFWarning, match="11.6.1"):
        keywords = XISF(path).get_images_metadata()[0]["FITSKeywords"]
    assert len(keywords) == 1
    (only,) = keywords.values()
    assert only == [{"value": "1", "comment": "c"}]


def test_missing_keyword_name_is_reported(tmp_path):
    """The name attribute is mandatory, so its absence is reported."""
    path = keyword_file(tmp_path, ['<FITSKeyword value="1" comment="c"/>'])
    with pytest.warns(XISFWarning):
        XISF(path).get_images_metadata()


# --------------------------------------------------------------------------
# Missing value or comment: a warning and an empty string
# --------------------------------------------------------------------------


def test_missing_comment_warns_and_defaults(tmp_path):
    """Regression test: a missing comment raised KeyError: 'comment'."""
    with pytest.warns(XISFWarning, match="comment"):
        XISF.write(
            str(tmp_path / "kw.xisf"),
            sample_image(),
            "test",
            {"FITSKeywords": {"EXPTIME": [{"value": "300"}]}},
        )
    assert read_keywords(str(tmp_path / "kw.xisf"))["EXPTIME"] == [
        {"value": "300", "comment": ""}
    ]


def test_missing_value_warns_and_defaults(tmp_path):
    """Regression test: a missing value raised KeyError: 'value'."""
    with pytest.warns(XISFWarning, match="value"):
        XISF.write(
            str(tmp_path / "kw.xisf"),
            sample_image(),
            "test",
            {"FITSKeywords": {"EXPTIME": [{"comment": "seconds"}]}},
        )
    assert read_keywords(str(tmp_path / "kw.xisf"))["EXPTIME"] == [
        {"value": "", "comment": "seconds"}
    ]


def test_entry_that_is_not_a_dict_is_reported(tmp_path):
    with pytest.warns(XISFWarning):
        XISF.write(
            str(tmp_path / "kw.xisf"),
            sample_image(),
            "test",
            {"FITSKeywords": {"EXPTIME": ["300"]}},
        )


@pytest.mark.parametrize("keyword", ["HISTORY", "COMMENT"])
def test_missing_value_is_expected_for_history_and_comment(tmp_path, keyword):
    """Those keywords have no value, so an empty value is not an anomaly.

    This is the case astropy produces when converting a FITS header, since
    HISTORY and COMMENT cards carry a comment and no value.
    """
    with _no_warnings():
        XISF.write(
            str(tmp_path / "kw.xisf"),
            sample_image(),
            "test",
            {"FITSKeywords": {keyword: [{"comment": "Processed"}]}},
        )
    assert read_keywords(str(tmp_path / "kw.xisf"))[keyword] == [
        {"value": "", "comment": "Processed"}
    ]


def test_history_and_comment_never_need_a_value(tmp_path):
    path = keyword_file(
        tmp_path, ['<FITSKeyword name="HISTORY" comment="Processed"/>']
    )
    with _no_warnings():
        keywords = XISF(path).get_images_metadata()[0]["FITSKeywords"]
    assert keywords["HISTORY"] == [{"value": "", "comment": "Processed"}]


# --------------------------------------------------------------------------
# Reading malformed attributes
# --------------------------------------------------------------------------


def test_reader_reports_missing_comment(tmp_path):
    """Regression test: reading a file without a comment raised KeyError."""
    path = keyword_file(tmp_path, ['<FITSKeyword name="EXPTIME" value="300"/>'])
    with pytest.warns(XISFWarning, match="comment"):
        keywords = XISF(path).get_images_metadata()[0]["FITSKeywords"]
    assert keywords["EXPTIME"] == [{"value": "300", "comment": ""}]


def test_reader_reports_missing_value(tmp_path):
    path = keyword_file(tmp_path, ['<FITSKeyword name="EXPTIME" comment="s"/>'])
    with pytest.warns(XISFWarning, match="value"):
        keywords = XISF(path).get_images_metadata()[0]["FITSKeywords"]
    assert keywords["EXPTIME"] == [{"value": "", "comment": "s"}]


def test_reader_keeps_the_rest_of_the_unit_accessible(tmp_path):
    """One bad keyword must not make the file unreadable (section 7)."""
    path = keyword_file(
        tmp_path,
        [
            '<FITSKeyword name="EXPTIME" value="300"/>',
            '<FITSKeyword name="EXPTIME" value="600" comment="second"/>',
        ],
    )
    with pytest.warns(XISFWarning):
        keywords = XISF(path).get_images_metadata()[0]["FITSKeywords"]
    assert len(keywords["EXPTIME"]) == 2


# --------------------------------------------------------------------------
# Non-string values and comments are converted
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,comment,absent", [(300, 1.5, None), (300, None, "comment"), (None, 1.5, "value")]
)
def test_non_string_value_or_comment_is_converted(tmp_path, value, comment, absent):
    """The XML attributes are text, so a number is stringified, not rejected."""
    entry = {}
    if value is not None:
        entry["value"] = value
    if comment is not None:
        entry["comment"] = comment
    # The absent attribute is reported by the writer, so the warning is asserted
    # here rather than being left to show up in the suite summary
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        XISF.write(
            str(tmp_path / "kw.xisf"),
            sample_image(),
            "test",
            {"FITSKeywords": {"EXPTIME": [entry]}},
        )
    reported = [str(m.message) for m in caught if issubclass(m.category, XISFWarning)]
    if absent is None:
        assert not reported, f"unexpected warnings: {reported}"
    else:
        assert len(reported) == 1
        assert f"no {absent} attribute" in reported[0]
    got = read_keywords(str(tmp_path / "kw.xisf"))["EXPTIME"][0]
    assert got["value"] == ("" if value is None else str(value))
    assert got["comment"] == ("" if comment is None else str(comment))


def test_wellformed_keywords_produce_no_warning(tmp_path):
    path = keyword_file(
        tmp_path,
        [
            '<FITSKeyword name="EXPTIME" value="300" comment="seconds"/>',
            '<FITSKeyword name="HISTORY" value="" comment="done"/>',
        ],
    )
    with _no_warnings():
        keywords = XISF(path).get_images_metadata()[0]["FITSKeywords"]
    assert keywords["EXPTIME"] == [{"value": "300", "comment": "seconds"}]
    assert keywords["HISTORY"] == [{"value": "", "comment": "done"}]
