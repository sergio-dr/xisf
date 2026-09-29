# coding: utf-8

"""
XISF Encoder/Decoder (see https://pixinsight.com/xisf/).

This implementation is not endorsed nor related with PixInsight development team.

Copyright (C) 2021-2026 Sergio Díaz, sergiodiaz.eu

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from importlib.metadata import version
__version__ = version(__name__)

import hashlib
import platform
import re
import xml.etree.ElementTree as ET
import numpy as np
import lz4.block  # https://python-lz4.readthedocs.io/en/stable/lz4.block.html
import zlib  # https://docs.python.org/3/library/zlib.html
import zstandard  # https://python-zstandard.readthedocs.io/en/stable/
import base64
import sys
import warnings
from datetime import datetime, timezone
import ast


class XISFWarning(UserWarning):
    """Warning category for XISF files that deviate from the specification or
    contain constructs this implementation cannot handle."""


class XISFError(ValueError):
    """Raised when a XISF file omits an attribute that the XISF 1.0 specification
    declares mandatory, leaving the decoder with no way to continue.

    This subclasses ValueError so that callers who already catch malformed-input
    errors around XISF keep working.
    """


class XISF:
    """Implements an baseline XISF Decoder and a simple baseline Encoder.
    It parses metadata from Image and Metadata XISF core elements. Image data is returned as a numpy ndarray
    (using the "channels-last" convention by default).

    What's supported:
    - Monolithic XISF files only
        - XISF data blocks with attachment, inline or embedded block locations
        - Both pixel storage models (planar and normal), for images of any dimensionality N >= 1
        - Data blocks in both byte orders, little- and big-endian (spec 10.4)
        - UInt8/16/32 and Float32/64 pixel sample formats
        - Grayscale and RGB color spaces
    - Decoding:
        - multiple Image core elements from a monolithic XISF file
        - Support all standard compression codecs defined in this specification for decompression
          (zlib/lz4[hc]/zstd + byte shuffling)
        - Verification of the SHA-1, SHA-256 and SHA-512 checksums that a baseline
          decoder shall support (spec 7.2), and of the optional SHA3-256 and
          SHA3-512 ones. A compressed block is verified before it is decompressed
          (spec 10.6.1).
    - Encoding:
        - Single image core element with an attached data block
        - Support all standard compression codecs defined in this specification for decompression
          (zlib/lz4[hc]/zstd + byte shuffling)
        - SHA-1 checksums of every data block by default, with any of the five
          algorithms of spec 10.5 available and checksums disableable
    - "Atomic" properties (scalar types, String, TimePoint), Vector and Matrix (e.g. astrometric
      solutions)
    - Metadata and FITSKeyword core elements

    What's not supported (at least by now):
    - Complex and Table properties
    - Any other not explicitly supported core elements (Resolution, Thumbnail, ICCProfile, etc.)

    Usage example:
    ```
    from xisf import XISF
    import matplotlib.pyplot as plt
    xisf = XISF("file.xisf")
    file_meta = xisf.get_file_metadata()
    file_meta
    ims_meta = xisf.get_images_metadata()
    ims_meta
    im_data = xisf.read_image(0)
    plt.imshow(im_data)
    plt.show()
    XISF.write(
        "output.xisf", im_data,
        creator_app="My script v1.0", image_metadata=ims_meta[0], xisf_metadata=file_meta,
        codec='lz4hc', shuffle=True
    )
    ```

    If the file is not huge and it contains only an image (or you're interested just in one of the
    images inside the file), there is a convenience method for reading the data and the metadata:
    ```
    from xisf import XISF
    import matplotlib.pyplot as plt
    im_data = XISF.read("file.xisf")
    plt.imshow(im_data)
    plt.show()
    ```

    The XISF format specification is available at https://pixinsight.com/doc/docs/XISF-1.0-spec/XISF-1.0-spec.html
    """

    # Static attributes
    _creator_app = f"Python {platform.python_version()}"
    _creator_module = f"XISF Python Module v{__version__} github.com/sergio-dr/xisf"
    _signature = b"XISF0100"  # Monolithic
    _headerlength_len = 4
    _reserved_len = 4
    _xml_ns = {"xisf": "http://www.pixinsight.com/xisf"}
    _xisf_attrs = {
        "xmlns": "http://www.pixinsight.com/xisf",
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "version": "1.0",
        "xsi:schemaLocation": "http://www.pixinsight.com/xisf http://pixinsight.com/xisf/xisf-1.0.xsd",
    }
    _compression_def_level = {
        "zlib": 6,  # 1..9, default: 6 as indicated in https://docs.python.org/3/library/zlib.html
        "lz4": 0,  # no other values, as indicated in https://python-lz4.readthedocs.io/en/stable/lz4.block.html
        "lz4hc": 9,  # 1..12, (4-9 recommended), default: 9 as indicated in https://python-lz4.readthedocs.io/en/stable/lz4.block.html
        "zstd": 3,  # 1..22, (3-9 recommended), default: 3 as indicated in https://facebook.github.io/zstd/zstd_manual.html
    }
    _block_alignment_size = 4096
    _max_inline_block_size = 3072

    def __init__(self, fname, verify_checksums=True):
        """Opens a XISF file and extract its metadata. To get the metadata and the images, see get_file_metadata(),
        get_images_metadata() and read_image().
        Args:
            fname: filename
            verify_checksums: whether to verify the checksum attribute of data blocks
              that declare one (spec 10.5). Verification happens when the block is
              read, not when the file is opened, and covers image and property
              data blocks alike. When a digest does not match, the block is not
              made available and XISFError is raised; set to False to downgrade
              the failure to an XISFWarning and return the data anyway. Passing
              False does not disable the check that a compressed block is never
              decompressed after a failed verification (spec 10.6.1).

        Returns:
            XISF object.
        """
        if not isinstance(verify_checksums, bool):
            raise XISFError(
                f"verify_checksums must be a bool, got"
                f" {type(verify_checksums).__name__}"
            )
        self._verify_checksums = verify_checksums
        self._fname = fname
        self._headerlength = None
        self._xisf_header = None
        self._xisf_header_xml = None
        self._images_meta = None
        self._images_xml = None
        self._file_meta = None
        ET.register_namespace("", self._xml_ns["xisf"])

        self._read()

    def _read(self):
        with open(self._fname, "rb") as f:
            # Check XISF signature
            signature = f.read(len(self._signature))
            if signature != self._signature:
                raise ValueError("File doesn't have XISF signature")

            # Get header length
            self._headerlength = int.from_bytes(f.read(self._headerlength_len), byteorder="little")
            # Equivalent:
            # self._headerlength = np.fromfile(f, dtype=np.uint32, count=1)[0]

            # Skip reserved field
            _ = f.read(self._reserved_len)

            # Get XISF (XML) Header
            #   https://github.com/sergio-dr/xisf/issues/6: some files have null padding at the 
            #   end of the header, so we strip it before parsing the XML
            self._xisf_header = f.read(self._headerlength).rstrip(b"\0")
            self._xisf_header_xml = ET.fromstring(self._xisf_header)
        self._analyze_header()

    def _analyze_header(self):
        # Analyze header to get Data Blocks position and length
        self._images_meta = []
        # The XML element of each image, index-aligned with _images_meta, needed to
        # read 'inline' and 'embedded' data blocks, whose contents are in the XML
        self._images_xml = []
        for image in self._xisf_header_xml.findall("xisf:Image", self._xml_ns):
            image_basic_meta = image.attrib

            # Parse and replace geometry and location with tuples,
            # parses and translates sampleFormat to numpy dtypes,
            # and extend with metadata from children entities (FITSKeywords, XISFProperties)

            #   The same FITS keyword can appear multiple times, so we have to
            #   prepare a dict of lists. Each element in the list is a dict
            #   that hold the value and the comment associated with the keyword.
            #   Not as clear as I would like.
            # The spec declares these Image attributes mandatory (section 12.1.1).
            # Checked before FITSKeyword parsing so a malformed Image is reported
            # on its own terms rather than as a missing FITS keyword.
            for attr in ("geometry", "location", "sampleFormat"):
                if attr not in image.attrib:
                    raise XISFError(
                        f"Image {image.attrib.get('id', '<unknown>')} is missing its"
                        f" mandatory '{attr}' attribute (required by the XISF 1.0"
                        f" spec, section 12.1.1)"
                    )

            fits_keywords = {}
            for a in image.findall("xisf:FITSKeyword", self._xml_ns):
                # The name, value and comment attributes are mandatory
                # (section 11.6.1). An encoder must not produce a unit without
                # them, but a decoder must keep the rest of the unit
                # accessible, so a missing one is reported and defaulted here
                # rather than raised.
                name = self._validate_fits_keyword_name(
                    a.attrib.get("name", ""), writing=False
                )
                value = a.attrib.get("value")
                if value is None:
                    if name not in self._fits_keywords_without_value:
                        self._warn_fits_keyword(
                            name,
                            "value",
                            "the value attribute is mandatory, so it is read as"
                            " an empty string",
                        )
                    value = ""
                comment = a.attrib.get("comment")
                if comment is None:
                    self._warn_fits_keyword(
                        name,
                        "comment",
                        "the comment attribute is mandatory, so it is read as an"
                        " empty string",
                    )
                    comment = ""
                fits_keywords.setdefault(name, []).append(
                    {"value": value.strip("'").strip(" "), "comment": comment}
                )

            image_extended_meta = {
                "geometry": self._parse_geometry(image.attrib["geometry"]),
                "location": self._parse_location(image.attrib["location"]),
                "dtype": self._parse_sampleFormat(image.attrib["sampleFormat"]),
                "FITSKeywords": fits_keywords,
                "XISFProperties": self._collect_properties(
                    image.findall("xisf:Property", self._xml_ns),
                    f"image {image.attrib.get('id', '<unknown>')}",
                ),
            }
            # Also parses compression attribute if present, converting it to a tuple
            if "compression" in image.attrib:
                image_extended_meta["compression"] = self._parse_compression(
                    image.attrib["compression"]
                )

            # pixelStorage is optional, and the default for an Image element
            # without it is the planar model (spec 11.5.2). A baseline decoder
            # shall read pixel data in both the planar and normal models
            # (spec 7.2), so it is recorded here for read_image to honor.
            image_extended_meta["pixelStorage"] = image.attrib.get(
                "pixelStorage", "Planar"
            )

            # The byteOrder attribute declares the endianness of the serialized
            # block, and little-endian is assumed when it is absent (spec 10.4).
            # A baseline decoder shall read data blocks in both byte orders
            # (spec 7.2), so it is recorded here for read_image to honor. It
            # applies to the uncompressed data of a compressed block as well.
            image_extended_meta["byteOrder"] = image.attrib.get(
                "byteOrder", "little"
            )

            # Merge basic and extended metadata in a dict
            image_meta = {**image_basic_meta, **image_extended_meta}

            # Append the image metadata to the list
            self._images_meta.append(image_meta)
            self._images_xml.append(image)

        # Analyze header for file metadata
        self._file_meta = {}
        metadata_elems = self._xisf_header_xml.findall("xisf:Metadata", self._xml_ns)
        if not metadata_elems:
            # The XISF 1.0 spec requires a unique Metadata element as a child of
            # the root element, so its absence is a deviation from the spec.
            warnings.warn(
                f"Missing <Metadata> element in XISF header of {self._fname}",
                XISFWarning,
            )
        elif len(metadata_elems) > 1:
            # Only one Metadata element is allowed; ignore any extra ones rather
            # than merging them, which would silently invent conflicting values.
            warnings.warn(
                f"Found {len(metadata_elems)} <Metadata> elements in XISF header of "
                f"{self._fname}; using the first",
                XISFWarning,
            )
        self._file_meta.update(
            self._collect_properties(
                metadata_elems[0] if metadata_elems else [], "the XISF unit"
            )
        )

        # TODO: rest of XISF core elements: Resolution, ICCProfile, Thumbnail, ...

    def _collect_properties(self, property_elems, owner):
        """Map Property elements to a dict keyed by id, warning on duplicates.

        A property identifier must be unique for the object with which the
        property is associated (spec 8.4.1). The specification does not say
        what a decoder must do when a file violates that requirement, so this
        keeps the last occurrence, which is what a dict assignment would do
        anyway, and reports the collision instead of dropping a value silently.

        Properties that cannot be parsed at all are skipped by
        _process_property(), which reports them itself, since a decoder must
        keep the rest of the unit accessible (spec 7). It signals that with a
        false value, which is why this tests for truthiness rather than None.
        """
        properties = {}
        for p in property_elems:
            prop = self._process_property(p)
            if not prop:
                continue
            prop_id = p.attrib["id"]
            if prop_id in properties:
                warnings.warn(
                    f"Found more than one property with id {prop_id!r} in"
                    f" {owner}; a property identifier must be unique for the"
                    f" object with which it is associated (XISF 1.0 spec,"
                    f" section 8.4.1). Keeping the last one.",
                    XISFWarning,
                )
            properties[prop_id] = prop
        return properties

    def get_images_metadata(self):
        """Provides the metadata of all image blocks contained in the XISF File, extracted from
        the header (<Image> core elements). To get the actual image data, see read_image().

        It outputs a dictionary m_i for each image, with the following structure:
        ```
        m_i = {
            'geometry': (dim1, ..., dimN, channels), # N >= 1; the trailing item is always the channel count
            'location': (pos, size), # used internally in read_image()
            'dtype': np.dtype('...'), # derived from sampleFormat argument
            'compression': (codec, uncompressed_size, item_size), # optional
            'key': 'value', # other <Image> attributes are simply copied
            ...,
            'FITSKeywords': { <fits_keyword>: fits_keyword_values_list, ... },
            'XISFProperties': { <xisf_property_name>: property_dict, ... }
        }

        where:

        fits_keyword_values_list = [ {'value': <value>, 'comment': <comment> }, ...]
        property_dict = {'id': <xisf_property_name>, 'type': <xisf_type>, 'value': property_value, ...}
        ```

        Returns:
            list [ m_0, m_1, ..., m_{n-1} ] where m_i is a dict as described above.

        """
        return self._images_meta

    def get_file_metadata(self):
        """Provides the metadata from the header of the XISF File (<Metadata> core elements).

        Returns:
            dictionary with one entry per property: { <xisf_property_name>: property_dict, ... }
            where:
            ```
            property_dict = {'id': <xisf_property_name>, 'type': <xisf_type>, 'value': property_value, ...}
            ```

        """
        return self._file_meta

    def get_metadata_xml(self):
        """Returns the complete XML header as a xml.etree.ElementTree.Element object.

        Returns:
            xml.etree.ElementTree.Element: complete XML XISF header
        """
        return self._xisf_header_xml

    def _read_data_block(self, elem, xml_elem):
        # xml_elem is the XML element that serialized the data block; it is needed
        # by the 'inline' and 'embedded' locations, whose contents are stored in the
        # XML serialization and not in the parsed metadata dict (elem).
        method = elem["location"][0]
        if method == "inline":
            return self._read_inline_data_block(elem, xml_elem)
        elif method == "embedded":
            return self._read_embedded_data_block(elem, xml_elem)
        elif method == "attachment":
            return self._read_attached_data_block(elem)
        else:
            raise NotImplementedError(f"Data block location type '{method}' not implemented: {elem}")

    def _verify_data_block_checksum(self, data, elem, owner):
        """Verify the checksum attribute of a data block, if it declares one.

        A decoder shall verify the checksum when the attribute is present
        (spec 10.5), and the verification takes place before the block is made
        available to the caller. This is called on the serialized block, before
        decompression, so that altered compressed data is never decompressed
        (spec 10.6.1).
        """
        if "checksum" not in elem:
            return
        algorithm, expected = self._parse_checksum(elem["checksum"], owner)
        computed = self._compute_checksum(algorithm, data)
        if computed == expected:
            return
        message = (
            f"{owner} fails checksum verification: it declares"
            f" {algorithm}:{expected} but its data block digests to"
            f" {algorithm}:{computed}. The data has been altered or corrupted"
            f" (XISF 1.0 spec, section 10.5)"
        )
        if not self._verify_checksums:
            warnings.warn(
                f"{message}. Returning the data anyway, since this XISF object"
                f" was opened with verify_checksums=False",
                XISFWarning,
            )
            return
        raise XISFError(message)

    def _block_owner(self, elem):
        """A human-readable name for the element that serialized a data block."""
        kind = elem.get("type") and "Property" or "Image"
        return f"{kind} {elem.get('id', '<unknown>')}"

    def _read_inline_data_block(self, elem, xml_elem):
        method, encoding = elem["location"]
        assert method == "inline"
        # An inline data block is serialized in the character contents of the element
        return self._decode_inline_or_embedded_data(encoding, xml_elem.text, elem)

    def _read_embedded_data_block(self, elem, xml_elem):
        assert elem["location"][0] == "embedded"
        # An embedded data block is serialized in the character contents of a child
        # Data element, which also carries the encoding and compression attributes
        data_elem = xml_elem.find("xisf:Data", XISF._xml_ns)
        if data_elem is None:
            raise ValueError(f"Embedded data block with no child Data element: {elem}")
        data_dict = dict(elem)
        if "compression" in data_elem.attrib:
            data_dict["compression"] = XISF._parse_compression(
                data_elem.attrib["compression"]
            )
        # The checksum attribute belongs to the element that serializes the block,
        # not to the child Data element (spec 10.5), so it is taken from elem
        return self._decode_inline_or_embedded_data(
            data_elem.attrib["encoding"], data_elem.text, data_dict
        )

    def _decode_inline_or_embedded_data(self, encoding, data, elem):
        # Base16 data is serialized in lowercase (see the XISF 1.0 spec), so
        # casefold=True is required to accept it
        encodings = {
            "base64": base64.b64decode,
            "hex": lambda d: base64.b16decode(d, casefold=True),
        }
        if encoding not in encodings:
            raise NotImplementedError(
                f"Data block encoding type '{encoding}' not implemented: {elem}"
            )

        data = encodings[encoding](data)
        # The digest is taken over the decoded binary data, not over its Base64 or
        # Base16 text (spec 10.5), and before decompression, since a compressed
        # block is verified as the compressed data (spec 10.6.1)
        self._verify_data_block_checksum(data, elem, self._block_owner(elem))
        if "compression" in elem:
            data = XISF._decompress(data, elem)

        return data

    def _read_attached_data_block(self, elem):
        # Position and size of the Data Block containing the image data
        method, pos, size = elem["location"]

        assert method == "attachment"

        with open(self._fname, "rb") as f:
            f.seek(pos)
            data = f.read(size)

        # Verified as the serialized block, before decompression (spec 10.6.1)
        self._verify_data_block_checksum(data, elem, self._block_owner(elem))
        if "compression" in elem:
            data = XISF._decompress(data, elem)

        return data

    def read_image(self, n=0, data_format="channels_last"):
        """Extracts an image from a XISF object.

        Images of any dimensionality N >= 1 are supported. The geometry attribute
        is dim1:...:dimN:channel-count, so the trailing item is always the channel
        count and never a spatial dimension (spec 11.5.1).

        The returned array is a read-only view on the decoded data, not a copy:
        it does not own its memory and cannot be modified in place. This is
        deliberate, since copying the array costs more than reading it (a
        256 MB image measures 125 ms to read against 154 ms extra to copy). To
        obtain a writable array, copy it, either with ndarray.copy() or with
        np.array(result).

        The data block byte order is honored (spec 10.4), so blocks in either
        byte order are decoded, and the returned array is always in the machine's
        byte order. A block needing a byte swap is a copy rather than a view, and
        is therefore writable.

        Note that the two data_format values do not have the same memory
        layout. Planar pixel storage decodes to (channels, *dims) in memory, so
        'channels_first' returns a C-contiguous array while 'channels_last' is a
        transposed view and is not C-contiguous. If you need the writable copy
        to be C-contiguous as well, use .copy() or np.array(result, order='C'),
        since np.array() alone defaults to order='K' and preserves the
        transposed layout.

        Args:
            n: index of the image to extract in the list returned by get_images_metadata()
            data_format: channels axis can be 'channels_first' or 'channels_last' (as used in
            keras/tensorflow, pyplot's imshow, etc.), 0 by default.

        Returns:
            Read-only Numpy ndarray with the image data, in the requested format
            (channels_first or channels_last). The shape is (dim1, ..., dimN,
            channels) for channels_last and (channels, dim1, ..., dimN) for
            channels_first. The array does not own its memory; see the note above
            if you need a writable one.

        """
        try:
            meta = self._images_meta[n]
        except IndexError as e:
            if self._xisf_header is None:
                raise RuntimeError("No file loaded") from e
            elif not self._images_meta:
                raise ValueError("File does not contain image data") from e
            else:
                raise ValueError(
                    f"Requested image #{n}, valid range is [0..{len(self._images_meta) - 1}]"
                ) from e

        # geometry is dim1:...:dimN:channel-count, so the channel count is the
        # trailing item and the spatial dimensions precede it (spec 11.5.1). dim1
        # is the X-axis, which is the last numpy axis, so the spatial dimensions
        # are reversed back into numpy array order.
        geometry = meta["geometry"]
        channels = geometry[-1]
        dims = geometry[:-1][::-1]

        data = self._read_data_block(meta, self._images_xml[n])
        dtype = self._dtype_for_byte_order(meta["dtype"], meta["byteOrder"])
        im_data = np.frombuffer(data, dtype=dtype)
        # Decode according to the pixel storage model: planar stores each channel
        # as a contiguous run of samples, while normal stores the samples of each
        # pixel together (spec 8.5.3). The default is the planar model
        # (spec 11.5.2). Both are read as (channels, *dims) here, since that is
        # the memory order the two models have in common.
        if meta.get("pixelStorage", "Planar") == "Planar":
            im_data = im_data.reshape((channels, *dims))
        else:
            im_data = np.moveaxis(im_data.reshape((*dims, channels)), -1, 0)
        if data_format == "channels_last":
            im_data = np.moveaxis(im_data, 0, -1)
        # The block was decoded with the dtype that interprets the serialized
        # bytes, which for a non-native byteOrder is not the native dtype.
        # Convert it, so the caller always receives an array in the machine's
        # byte order. This allocates a new array, which is unavoidable since the
        # block is a read-only view, and it also makes the result writable.
        if dtype != meta["dtype"]:
            im_data = im_data.astype(meta["dtype"])
        return im_data

    @staticmethod
    def read(fname, n=0, image_metadata=None, xisf_metadata=None):
        """Convenience method for reading a file containing a single image.

        Args:
            fname (string): filename
            n (int, optional): index of the image to extract (in the list returned by get_images_metadata()). Defaults to 0.
            image_metadata (dict, optional): dictionary that will be updated with the metadata of the image.
              If None, the image metadata is not collected, since the caller has
              nowhere to receive it. Defaults to None.
            xisf_metadata (dict, optional): dictionary that will be updated with the metadata of the file.
              If None, the file metadata is not collected, since the caller has
              nowhere to receive it. Defaults to None.

        Returns:
            [np.ndarray]: Numpy ndarray with the image data, in the requested format (channels_first or channels_last).
            The array is a read-only view on the decoded data and cannot be modified in place;
              pass it to np.array() or call .copy() if you need a writable array, as read_image() does.
        """
        xisf = XISF(fname)
        # These are only output parameters, so there is nothing to collect into
        # when the caller has not provided a dictionary. Their previous default
        # was a shared mutable dict, which accumulated the metadata of every
        # file read without ever being visible to the caller.
        if xisf_metadata is not None:
            xisf_metadata.update(xisf.get_file_metadata())
        if image_metadata is not None:
            image_metadata.update(xisf.get_images_metadata()[n])
        return xisf.read_image(n)

    # if 'colorSpace' is not specified, im_data.shape[2] dictates if colorSpace is 'Gray' or 'RGB'
    @staticmethod
    def _resolve_image_data(im_data, pixel_storage=None):
        """Split an image array into its spatial dimensions and channel count.

        The XISF geometry attribute is dim1:...:dimN:channel-count, where N >= 1
        is the dimensionality of the image and the channel count is always a
        separate trailing item, never one of the dimensions (spec 11.5.1). A
        numpy array carries no such distinction, so the channel axis is inferred:

        Args:
            im_data: an array of at least one dimension. A 1-D or 2-D array is
                always single-channel. A 3-D array is taken as
                (dim1, dim2, channels) for the normal model when the trailing
                axis is 1 or 3, otherwise as (channels, dim1, dim2) for the
                planar model when the leading axis is 1 or 3, and otherwise as
                three single-channel dimensions. An array of four or more axes
                is always taken as purely spatial.
            pixel_storage: 'normal', 'planar', or None to infer the layout from
                the shape. When given explicitly, the array must be at least
                2-D, and its channel axis is the trailing axis for 'normal' or
                the leading axis for 'planar'.

        Returns:
            The spatial dimension lengths, the channel count, and the storage
            model assumed for the input. The array is not transposed here; the
            caller reorders it with _reorder_to_dims.
        """
        if not isinstance(im_data, np.ndarray):
            raise XISFError(
                f"im_data must be a numpy ndarray, got {type(im_data).__name__}"
            )
        if im_data.ndim < 1:
            # geometry is dim1:...:dimN:channel-count with N >= 1, so an image
            # must have at least one dimension besides its channel count
            raise XISFError(
                f"im_data must have at least one dimension, got a 0-D array"
            )
        if 0 in im_data.shape:
            # every dimi item shall be greater than zero (spec 11.5.1)
            raise XISFError(
                f"Every geometry dimension shall be greater than zero (spec"
                f" 11.5.1), got an array of shape {im_data.shape}"
            )

        if pixel_storage is None:
            if im_data.ndim == 3:
                if im_data.shape[2] in (1, 3):
                    # Retain the historical inference so existing callers are
                    # unaffected: a trailing 1 or 3 is the channel count.
                    return im_data.shape[:2], im_data.shape[2], "normal"
                if im_data.shape[0] in (1, 3):
                    return im_data.shape[1:], im_data.shape[0], "planar"
            # One or two axes, or an array with no axis that could plausibly be
            # a channel count, is a single-channel image of that dimensionality.
            return im_data.shape, 1, "planar"

        if not isinstance(pixel_storage, str):
            raise XISFError(
                f"pixel_storage must be 'planar' or 'normal', got"
                f" {pixel_storage!r}"
            )
        pixel_storage = pixel_storage.lower()

        if pixel_storage == "normal":
            if im_data.ndim < 2:
                raise XISFError(
                    f"pixel_storage='normal' expects a trailing channel axis, so"
                    f" im_data must have at least two dimensions, got shape"
                    f" {im_data.shape}"
                )
            return im_data.shape[:-1], im_data.shape[-1], "normal"
        if pixel_storage == "planar":
            if im_data.ndim < 2:
                raise XISFError(
                    f"pixel_storage='planar' expects a leading channel axis, so"
                    f" im_data must have at least two dimensions, got shape"
                    f" {im_data.shape}"
                )
            return im_data.shape[1:], im_data.shape[0], "planar"
        raise XISFError(
            f"pixel_storage must be 'planar' or 'normal', got {pixel_storage!r}"
        )

    @staticmethod
    def _reorder_to_dims(im_data, dims, channels, storage):
        """Reorder an image array to (channels, *dims) as the planar model stores it.

        In the planar model each channel is stored as a contiguous sequence of
        pixel samples, and channels are stored consecutively in increasing order
        of channel index (spec 8.5.3.1). The samples of a channel are stored in
        pixel coordinate order with the first coordinate varying fastest, which
        is exactly the memory order of a C-contiguous (channels, *dims) array.

        Args:
            im_data: the input array.
            dims: spatial dimension lengths, in axis order.
            channels: the channel count.
            storage: 'normal' for a trailing channel axis, 'planar' for a leading one.
        """
        if storage == "normal":
            # move the trailing channel axis to the front
            im_data = np.moveaxis(im_data, -1, 0)
        return np.ascontiguousarray(im_data).reshape((channels, *dims))

    @staticmethod
    def _color_space_for_channels(channels):
        """Return the XISF colorSpace literal for a channel count.

        Table 14 in spec 11.5.2 permits exactly three literals, Gray, RGB and
        CIELab, and RGB and CIELab are each defined over three nominal channels.
        That does not constrain the total channel count: spec 8.5.1 calls the
        first channels strictly required to define the color model the nominal
        channels and everything beyond them alpha channels, and 11.5.1 requires
        only that the channel count be greater than zero. So any other count is
        valid, and the nominal channels are grayscale with the remainder as
        alpha channels. Gray is the default assumed for an image without a
        colorSpace attribute (11.5.2), so it is also the right literal here.
        """
        if channels == 3:
            return "RGB"
        return "Gray"

    # For float sample formats, bounds="0:1" is assumed
    @staticmethod
    def write(
        fname,
        im_data,
        creator_app=None,
        image_metadata=None,
        xisf_metadata=None,
        codec=None,
        shuffle=False,
        level=None,
        pixel_storage=None,
        checksum="sha-1",
    ):
        """Writes an image (numpy array) to a XISF file. Compression may be requested but it only
        will be used if it actually reduces the data size.

        Args:
            fname: filename (will overwrite if existing)
            im_data: numpy ndarray with the image data. Arrays of any dimensionality
              N >= 1 are written, with the geometry written as
              dim1:...:dimN:channel-count (spec 11.5.1). A 1-D or 2-D array is
              single-channel. A 3-D array is interpreted according to pixel_storage, so
              pass it explicitly if the shape is ambiguous. An array of four or more axes
              is written as purely spatial dimensions. The array shall be in the
              machine's byte order, which is the byte order written to the file
              (spec 10.4); an array with an explicit foreign byte order raises
              XISFError rather than being written with a byteOrder attribute that
              misdescribes it. Convert it with, for example,
              im_data.astype(im_data.dtype.newbyteorder("=")).
            creator_app: string for XISF:CreatorApplication file property (defaults to python version in None provided)
            image_metadata: dict with the same structure described for m_i in get_images_metadata().
              Only 'id', 'FITSKeywords' and 'XISFProperties' keys are actually written. The 'id'
              defaults to 'image' and shall match [_a-zA-Z][_a-zA-Z0-9]* (spec 11.5.2), an
              invalid one raises XISFError. The rest of the keys, such as 'geometry',
              'sampleFormat' and 'pixelStorage', are derived from im_data and ignored.
            xisf_metadata: file metadata, dict with the same structure returned by get_file_metadata()
            codec: compression codec ('zlib', 'lz4', 'lz4hc' or 'zstd'), or None to disable compression
            shuffle: whether to apply byte-shuffling before compression (ignored if codec is None). Recommended
              for 'lz4' ,'lz4hc' and 'zstd' compression algorithms.
            level: for zlib, 1..9 (default: 6); for lz4hc, 1..12 (default: 9); for zstd, 1..22 (default: 3).
              Higher means more compression.
            pixel_storage: pixel storage model of im_data, using the spec naming
              (spec 8.5.3). 'planar' (channels first) means a (channels, *dims) array,
              which is the 'channels_first' layout used by keras and TensorFlow;
              'normal' (channels last) means a (*dims, channels) array, which is the
              'channels_last' layout used by numpy, matplotlib and most image tooling.
              When given, im_data must have at least two dimensions and the channel
              count is the leading axis for 'planar' or the trailing axis for 'normal'.
              Defaults to None, which infers the layout from the shape: a 3-D array
              with a trailing dimension of 1 or 3 is read as (dim1, dim2, channels),
              one with a leading dimension of 1 or 3 as (channels, dim1, dim2), and
              anything else is treated as single-channel spatial dimensions.
            checksum: the cryptographic hashing algorithm used for the checksum
              attribute of every data block written, one of 'sha-1' (the default,
              and the algorithm recommended by the spec), 'sha-256', 'sha-512',
              'sha3-256' or 'sha3-512'. The alternate spellings without the
              hyphen are accepted too. True is the same as 'sha-1', and False or
              None writes no checksums. The digest is computed for the serialized
              block, so for a compressed block it is the digest of the compressed
              data (spec 10.6.1), and the algorithms applied are listed in the
              XISF:ChecksumAlgorithms file property.
        Returns:
            bytes_written: the total number of bytes written into the output file.
            codec: The codec actually used, i.e., None if compression did not reduce the data block size so
              compression was not finally used.

        """
        checksum_algorithm = XISF._validate_checksum_algorithm(checksum)

        if image_metadata is None:
            image_metadata = {}

        if xisf_metadata is None:
            xisf_metadata = {}

        # Data block alignment
        blk_sz = xisf_metadata.get("XISF:BlockAlignmentSize", {"value": XISF._block_alignment_size})[
            "value"
        ]
        # Maximum inline block size (larger will be attached instead)
        max_inline_blk_sz = xisf_metadata.get(
            "XISF:MaxInlineBlockSize", {"value": XISF._max_inline_block_size}
        )["value"]

        # Split the array into spatial dimensions and a channel count before any
        # metadata is derived from it, so that geometry and the data block always
        # agree (spec 11.5.1: geometry is dim1:...:dimN:channel-count, N >= 1)
        dims, channels, input_storage = XISF._resolve_image_data(im_data, pixel_storage)

        # The data block is serialized as-is with tobytes(), so its byte order is
        # the byte order of im_data, and the byteOrder attribute declared for it is
        # derived from the machine. An array that is not already in the machine's
        # byte order would therefore be written with a byteOrder attribute that
        # misdescribes it, so it is rejected instead of silently corrupting the
        # file. Endianness is immaterial for single-byte sample formats.
        if im_data.dtype.itemsize > 1 and im_data.dtype.byteorder not in (
            "=",
            "|",
            sys.byteorder,
        ):
            raise XISFError(
                f"im_data has byte order '{im_data.dtype.str}', but the machine is"
                f" {sys.byteorder}-endian and the data block is written as-is."
                f" Convert it first, for example with"
                f" im_data.astype(im_data.dtype.newbyteorder('='))"
            )
        im_data = XISF._reorder_to_dims(im_data, dims, channels, input_storage)
        # geometry is dim1:...:dimN:channel-count (spec 11.5.1), where dim1 is the
        # X-axis and dimN the Y-axis for a 2-D image. A numpy array is indexed
        # rows-first, so dim1 is the *last* array axis. The spatial dimensions are
        # therefore emitted in reverse array order, and the data block keeps the
        # array order, in which the first array axis varies fastest.
        geometry = ":".join(str(dim) for dim in (*reversed(dims), channels))

        # Prepare basic image metadata
        def _create_image_metadata(id):
            image_attrs = {"id": id}
            image_attrs["geometry"] = geometry
            image_attrs["colorSpace"] = XISF._color_space_for_channels(channels)
            # A baseline encoder shall write pixel data in the planar storage
            # model (spec 7.1). It is also the default a decoder assumes when the
            # attribute is absent (spec 11.5.2), but it is written explicitly so
            # that the model is never ambiguous.
            image_attrs["pixelStorage"] = "Planar"
            image_attrs["sampleFormat"] = XISF._get_sampleFormat(im_data.dtype)
            if image_attrs["sampleFormat"].startswith("Float"):
                image_attrs["bounds"] = "0:1"  # Assumed
            # The data block is serialized as im_data.tobytes(), so it carries
            # whatever byte order im_data has, and the byteOrder attribute
            # declares that (spec 10.4). It is derived from the machine, since
            # _check_native_byte_order has already rejected any array that does
            # not already be in the machine's byte order, so the attribute is
            # only ever needed to spell out the little-endian default. It is
            # meaningless for single-byte sample formats, which have no byte
            # order to declare.
            if sys.byteorder == "big" and im_data.dtype.itemsize > 1:
                image_attrs["byteOrder"] = "big"
            return image_attrs

        # Serialize a data block, with optional compression (i.e., when codec is not None)
        # Compression will be only applied if effectively reduces size
        def _serialize_data_block(data, attr_dict, codec, level, shuffle):
            data_block = data.tobytes()
            uncompressed_size = data.nbytes
            codec_str = codec

            if codec is None:
                data_size = uncompressed_size
            else:
                compressed_block = XISF._compress(data_block, codec, level, shuffle, data.itemsize)
                compressed_size = len(compressed_block)

                if compressed_size < uncompressed_size:
                    # The ideal situation, compressing actually reduces size
                    data_block, data_size = compressed_block, compressed_size

                    # Add 'compression' image attribute: (codec:uncompressed-size[:item-size])
                    if shuffle:
                        codec_str += "+sh"
                        attr_dict["compression"] = f"{codec_str}:{uncompressed_size}:{data.itemsize}"
                    else:
                        attr_dict["compression"] = f"{codec}:{uncompressed_size}"
                else:
                    # If there's no gain in compressing, just discard the compressed block
                    # See https://pixinsight.com/forum.old/index.php?topic=10942.msg68043#msg68043
                    # (In fact, PixInsight will show garbage image data if the data block is
                    # compressed but the uncompressed size is smaller)
                    data_size = uncompressed_size
                    codec_str = None

            return data_block, data_size, codec_str

        # Overwrites/creates XISF metadata
        def _update_xisf_metadata(creator_app, blk_sz, max_inline_blk_sz, codec, level):
            # Create file metadata
            xisf_metadata["XISF:CreationTime"] = {
                "id": "XISF:CreationTime",
                "type": "String",
                "value": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            }
            xisf_metadata["XISF:CreatorApplication"] = {
                "id": "XISF:CreatorApplication",
                "type": "String",
                "value": creator_app if creator_app else XISF._creator_app,
            }
            xisf_metadata["XISF:CreatorModule"] = {
                "id": "XISF:CreatorModule",
                "type": "String",
                "value": XISF._creator_module,
            }
            _OSes = {
                "linux": "Linux",
                "win32": "Windows",
                "cygwin": "Windows",
                "darwin": "macOS",
            }
            xisf_metadata["XISF:CreatorOS"] = {
                "id": "XISF:CreatorOS",
                "type": "String",
                "value": _OSes[sys.platform],
            }
            xisf_metadata["XISF:BlockAlignmentSize"] = {
                "id": "XISF:BlockAlignmentSize",
                "type": "UInt16",
                "value": blk_sz,
            }
            xisf_metadata["XISF:MaxInlineBlockSize"] = {
                "id": "XISF:MaxInlineBlockSize",
                "type": "UInt16",
                "value": max_inline_blk_sz,
            }
            if codec is not None:
                # Add XISF:CompressionCodecs and XISF:CompressionLevel to file metadata
                xisf_metadata["XISF:CompressionCodecs"] = {
                    "id": "XISF:CompressionCodecs",
                    "type": "String",
                    "value": codec,
                }
                xisf_metadata["XISF:CompressionLevel"] = {
                    "id": "XISF:CompressionLevel",
                    "type": "Int",
                    "value": level if level else XISF._compression_def_level[codec],
                }
            else:
                # Remove compression metadata if exists
                try:
                    del xisf_metadata["XISF:CompressionCodecs"]
                    del xisf_metadata["XISF:CompressionLevel"]
                except:
                    pass

        def _compute_attached_positions(hdr_prov_sz, attached_blocks_locations):
            # Computes aligned position nearest to the given one
            _aligned_position = lambda pos: ((pos + blk_sz - 1) // blk_sz) * blk_sz

            # Iterates data block positions until header size stabilizes
            # (positions are represented as strings in the header so their
            # values may impact header size, therefore changing data block
            # positions in the file)
            hdr_sz = hdr_prov_sz
            prev_sum_len_positions = 0
            while True:
                # account for the size of the (provisional) header
                pos = _aligned_position(hdr_sz)

                # positions for data blocks of properties with attachment location
                sum_len_positions = 0
                for loc in attached_blocks_locations:
                    # Save the (possibly provisional) position
                    loc['position'] = pos
                    # Accumulate the size of the position string
                    sum_len_positions += len(str(pos))
                    # Fast forward position adding the size, honoring alignment
                    pos = _aligned_position(pos + loc['size'])

                if sum_len_positions == prev_sum_len_positions:
                    break

                prev_sum_len_positions = sum_len_positions
                hdr_sz = hdr_prov_sz + sum_len_positions

            # Update data blocks positions in XML Header
            for b in attached_blocks_locations:
                xml_elem, pos, sz = b["xml"], b["position"], b["size"]
                xml_elem.attrib["location"] = XISF._to_location(("attachment", pos, sz))

        # Zero padding (used for reserved fields and data block alignment)
        def _zero_pad(length):
            assert length >= 0
            return (0).to_bytes(length, byteorder="little")

        # Add the checksum attribute to every element that serializes a data
        # block. It is done in one pass over the assembled header rather than in
        # each of the branches that build a data block, so that image and
        # property blocks are all covered, and only for elements that actually
        # have one. The digest is taken over the serialized block: the decoded
        # binary data rather than its Base64 or Base16 text for an inline or
        # embedded block (spec 10.5), and the compressed data for a compressed
        # block (spec 10.6.1).
        def _add_checksums(header_xml, attached_blocks_locations, algorithm):
            if algorithm is None:
                return set()
            algorithms = set()

            def _digest_of(data):
                algorithms.add(algorithm)
                return f"{algorithm}:{XISF._compute_checksum(algorithm, data)}"

            # Attached blocks: their serialized bytes are the bytes to be written.
            # A Vector or Matrix property block is held as an ndarray and written
            # as its buffer, so it is converted the same way before hashing.
            for block in attached_blocks_locations:
                data = block["data"]
                if isinstance(data, np.ndarray):
                    data = data.tobytes()
                block["xml"].attrib["checksum"] = _digest_of(data)

            # Inline and embedded blocks: the serialized bytes are the element text
            # encoded, so they are recovered from it
            for elem in header_xml.iter():
                location = elem.attrib.get("location")
                if not location or not location.startswith("inline:"):
                    continue
                encoding = location.partition(":")[2]
                encodings = {
                    "base64": lambda t: base64.b64decode(t),
                    "hex": lambda t: base64.b16decode(t, casefold=True),
                }
                data = encodings[encoding](elem.text or "")
                elem.attrib["checksum"] = _digest_of(data)

            return algorithms

        # __/ Prepare image and its metadata \__________
        im_id = image_metadata.get("id", "image")
        XISF._validate_image_id(im_id)
        im_attrs = _create_image_metadata(im_id)
        im_data_block, data_size, codec_str = _serialize_data_block(
            im_data, im_attrs, codec, level, shuffle
        )

        # Assemble location attribute, *provisional* until we can compute the data block position
        im_attrs["location"] = XISF._to_location(("attachment", "", data_size))

        # __/ Build (provisional) XML Header \__________
        # (for attached data blocks, the location is provisional)
        #   Convert metadata (dict) to XML Header
        xisf_header_xml = ET.Element("xisf", XISF._xisf_attrs)

        #   Image
        image_xml = ET.SubElement(xisf_header_xml, "Image", im_attrs)

        #     Image FITSKeywords
        for kw_name, kw_values in image_metadata.get("FITSKeywords", {}).items():
            XISF._insert_fitskeyword(image_xml, kw_name, kw_values)

        # attached_blocks_locations will reference every element whose data block is to be attached
        #   = [{"xml": ElementTree, "position": int, "size": int, "data": ndarray or str}]
        #   (position key is actually a placeholder, it will be overwritten by
        #   _compute_attached_positions)
        # The first element is the image (*provisional* location):
        attached_blocks_locations = [
            {
                "xml": image_xml,
                "position": 0,
                "size": data_size,
                "data": im_data_block,
            }
        ]

        #     Image XISFProperties
        for p_dict in image_metadata.get("XISFProperties", {}).values():
            if attached_block := XISF._insert_property(image_xml, p_dict, max_inline_blk_sz):
                attached_blocks_locations.append(attached_block)

        #   File Metadata
        metadata_xml = ET.SubElement(xisf_header_xml, "Metadata")
        _update_xisf_metadata(creator_app, blk_sz, max_inline_blk_sz, codec, level)
        for property_dict in xisf_metadata.values():
            if attached_block := XISF._insert_property(
                metadata_xml, property_dict, max_inline_blk_sz
            ):
                attached_blocks_locations.append(attached_block)

        # Checksum every data block that has been written to the header, now that
        # the image and all the properties are assembled. Adding the attribute
        # changes the header size, so it is done before the provisional header
        # size is measured below.
        checksum_algorithms = _add_checksums(
            xisf_header_xml, attached_blocks_locations, checksum_algorithm
        )
        if checksum_algorithms:
            # If the unit contains data blocks with checksums, this property
            # should enumerate the applied algorithms (spec 11.2). It is added
            # after the checksums, since it is a String property serialized as
            # character contents rather than as a data block.
            ET.SubElement(
                metadata_xml,
                "Property",
                {
                    "id": "XISF:ChecksumAlgorithms",
                    "type": "String",
                },
            ).text = ",".join(sorted(checksum_algorithms))

        # Header provisional size (without attachment positions)
        xisf_header = ET.tostring(xisf_header_xml, encoding="utf8")
        header_provisional_sz = (
            len(XISF._signature) + XISF._headerlength_len + len(xisf_header) + XISF._reserved_len
        )

        # Update location for every block in attached_blocks_locations
        _compute_attached_positions(header_provisional_sz, attached_blocks_locations)

        with open(fname, "wb") as f:
            # Write XISF signature
            f.write(XISF._signature)

            xisf_header = ET.tostring(xisf_header_xml, encoding="utf8")
            headerlength = len(xisf_header)
            # Write header length
            f.write(headerlength.to_bytes(XISF._headerlength_len, byteorder="little"))

            # Write reserved field
            reserved_field = _zero_pad(XISF._reserved_len)
            f.write(reserved_field)

            # Write header
            f.write(xisf_header)

            # Write data blocks
            for b in attached_blocks_locations:
                pos, data_block = b["position"], b["data"]
                f.write(_zero_pad(pos - f.tell()))
                assert f.tell() == pos
                f.write(data_block)
            bytes_written = f.tell()

        return bytes_written, codec_str

    # __/ Auxiliary functions to handle XISF attributes \________

    # Process property attributes and convert to dict
    def _process_property(self, p_et):
        p_dict = p_et.attrib.copy()

        if p_dict["type"] == "TimePoint":
            # Timepoint 'value' attribute already set (as str)
            # TODO: convert to datetime?
            pass
        elif p_dict["type"] == "String":
            p_dict["value"] = p_et.text
            if "location" in p_dict:
                # Process location and compression attributes to find data block
                self._process_location_compression(p_dict)
                p_dict["value"] = self._read_data_block(p_dict, p_et).decode("utf-8")
        elif p_dict["type"] == "Boolean":
            p_dict["value"] = self._parse_boolean(p_dict)
        elif "Vector" in p_dict["type"]:
            # A Vector property shall not have a value attribute (spec 11.1.8),
            # so it must be tested before the scalar fallback below
            if "value" in p_et.attrib:
                warnings.warn(
                    f"Vector property {p_dict['id']} has a forbidden value attribute,"
                    f" ignoring it",
                    XISFWarning,
                )
            p_dict["length"] = self._require_int_attr(p_dict, "length", "11.1.8")
            p_dict["dtype"] = self._parse_vector_dtype(p_dict["type"])
            self._process_location_compression(p_dict)
            raw_data = self._read_data_block(p_dict, p_et)
            p_dict["value"] = np.frombuffer(raw_data, dtype=p_dict["dtype"], count=p_dict["length"])
        elif "Matrix" in p_dict["type"]:
            # A Matrix property shall not have a value attribute (spec 11.1.9),
            # so it must be tested before the scalar fallback below
            if "value" in p_et.attrib:
                warnings.warn(
                    f"Matrix property {p_dict['id']} has a forbidden value attribute,"
                    f" ignoring it",
                    XISFWarning,
                )
            p_dict["rows"] = self._require_int_attr(p_dict, "rows", "11.1.9")
            p_dict["columns"] = self._require_int_attr(p_dict, "columns", "11.1.9")
            length = p_dict["rows"] * p_dict["columns"]
            p_dict["dtype"] = self._parse_vector_dtype(p_dict["type"])
            self._process_location_compression(p_dict)
            raw_data = self._read_data_block(p_dict, p_et)
            p_dict["value"] = np.frombuffer(raw_data, dtype=p_dict["dtype"], count=length)
            p_dict["value"] = p_dict["value"].reshape((p_dict["rows"], p_dict["columns"]))
        elif "value" in p_et.attrib:
            # Scalars (Float64, UInt32, etc.) and Complex*
            p_dict["value"] = ast.literal_eval(p_dict["value"])
        else:
            warnings.warn(
                f"Unsupported Property type {p_dict['type']}: {p_et}",
                XISFWarning,
            )
            p_dict = False

        return p_dict

    @staticmethod
    def _parse_boolean(p_dict):
        """Return a Boolean property value as a Python bool.

        A serialization of a Boolean value as plain text shall be one of the
        words true and false, and decoders shall also accept the integers 1 and
        0 as serializations of true and false, respectively (section 8.3.4).
        Leading and trailing white space is irrelevant and must be ignored
        (section 8.3.5).

        The spec does not state whether the words are case-sensitive, so any
        casing is accepted here. This is a superset of the defined forms and
        keeps files written by other implementations, or by earlier versions of
        this package, which serialized True as "True", readable.
        """
        raw = p_dict["value"].strip()
        lowered = raw.lower()
        if lowered == "true" or raw == "1":
            return True
        if lowered == "false" or raw == "0":
            return False
        raise XISFError(
            f"Property {p_dict.get('id', '<unknown>')} of type Boolean has a"
            f" malformed value {raw!r}: expected 'true' or 'false', or the"
            f" integers 1 or 0 (required by the XISF 1.0 spec, section 8.3.4)"
        )

    @staticmethod
    def _require_int_attr(p_dict, name, spec_section):
        """Return a mandatory unsigned integer attribute as an int.

        The spec declares these attributes mandatory, so a missing one is
        unrecoverable: without the component count there is no way to know how
        much of the data block belongs to the property. A malformed value is
        equally unrecoverable, since it cannot be interpreted.
        """
        prop_id = p_dict.get("id", "<unknown>")
        if name not in p_dict:
            raise XISFError(
                f"Property {prop_id} of type {p_dict['type']} is missing its mandatory"
                f" '{name}' attribute (required by the XISF 1.0 spec, section"
                f" {spec_section})"
            )
        try:
            return int(p_dict[name])
        except (TypeError, ValueError) as e:
            raise XISFError(
                f"Property {prop_id} has a malformed '{name}' attribute:"
                f" {p_dict[name]!r} is not an integer (required by the XISF 1.0"
                f" spec, section {spec_section})"
            ) from e

    @staticmethod
    def _require_location(p_dict):
        """Return a mandatory location attribute, parsed.

        A Vector or Matrix shall have a location attribute (spec 11.1.8, 11.1.9),
        and its value is serialized as an XISF data block. Without it there is no
        data block to read, so the property cannot be decoded.
        """
        if "location" not in p_dict:
            raise XISFError(
                f"Property {p_dict.get('id', '<unknown>')} of type {p_dict['type']} is"
                f" missing its mandatory 'location' attribute, so its data block"
                f" cannot be found (required by the XISF 1.0 spec, section 11.1.8"
                f" and 11.1.9)"
            )
        return XISF._parse_location(p_dict["location"])

    @staticmethod
    def _process_location_compression(p_dict):
        p_dict["location"] = XISF._require_location(p_dict)
        if "compression" in p_dict:
            p_dict["compression"] = XISF._parse_compression(p_dict["compression"])

    # Insert XISF properties in the XML tree
    @staticmethod
    def _insert_property(parent, p_dict, max_inline_block_size):
        # TODO ignores optional attributes (format, comment)
        scalars = ["Int", "Byte", "Short", "Float", "Boolean", "TimePoint"]

        if any(t in p_dict["type"] for t in scalars):
            # scalars and TimePoint
            # TODO add check for scalar or TimePoint
            # The Boolean literals are lowercase in the spec (section 11.1.4),
            # but str(True) is "True", so they are formatted explicitly.
            if p_dict["type"] == "Boolean":
                value = "true" if p_dict["value"] else "false"
            else:
                value = str(p_dict["value"])
            ET.SubElement(
                parent,
                "Property",
                {
                    "id": p_dict["id"],
                    "type": p_dict["type"],
                    "value": value,
                },
            )
        elif p_dict["type"] == "String":
            text = str(p_dict["value"])
            sz = len(text.encode("utf-8"))
            if sz > max_inline_block_size:
                # Attach string as data block (position pending)
                # TODO ignores compression
                xml = ET.SubElement(
                    parent,
                    "Property",
                    {
                        "id": p_dict["id"],
                        "type": p_dict["type"],
                        "location": XISF._to_location(("attachment", "", sz)),
                    },
                )
                return {"xml": xml, "location": 0, "size": sz, "data": text.encode()}
            else:
                # string directly as child (no 'location' attribute)
                ET.SubElement(
                    parent,
                    "Property",
                    {
                        "id": p_dict["id"],
                        "type": p_dict["type"],
                    },
                ).text = text
        elif "Vector" in p_dict["type"]:
            # TODO ignores compression
            data = p_dict["value"]
            sz = data.nbytes
            if sz > max_inline_block_size:
                # Attach vector as data block (position pending)
                xml = ET.SubElement(
                    parent,
                    "Property",
                    {
                        "id": p_dict["id"],
                        "type": p_dict["type"],
                        "length": str(data.size),
                        "location": XISF._to_location(("attachment", "", sz)),
                    },
                )
                return {"xml": xml, "location": 0, "size": sz, "data": data}
            else:
                # Inline data block (assuming base64)
                ET.SubElement(
                    parent,
                    "Property",
                    {
                        "id": p_dict["id"],
                        "type": p_dict["type"],
                        "length": str(data.size),
                        "location": XISF._to_location(("inline", "base64")),
                    },
                ).text = str(base64.b64encode(data.tobytes()), "ascii")
        elif "Matrix" in p_dict["type"]:
            # TODO ignores compression
            data = p_dict["value"]
            sz = data.nbytes
            if sz > max_inline_block_size:
                # Attach vector as data block (position pending)
                xml = ET.SubElement(
                    parent,
                    "Property",
                    {
                        "id": p_dict["id"],
                        "type": p_dict["type"],
                        "rows": str(data.shape[0]),
                        "columns": str(data.shape[1]),
                        "location": XISF._to_location(("attachment", "", sz)),
                    },
                )
                return {"xml": xml, "location": 0, "size": sz, "data": data}
            else:
                # Inline data block (assuming base64)
                ET.SubElement(
                    parent,
                    "Property",
                    {
                        "id": p_dict["id"],
                        "type": p_dict["type"],
                        "rows": str(data.shape[0]),
                        "columns": str(data.shape[1]),
                        "location": XISF._to_location(("inline", "base64")),
                    },
                ).text = str(base64.b64encode(data.tobytes()), "ascii")
        else:
            warnings.warn(
                f"Skipping unsupported property {p_dict}",
                XISFWarning,
            )

        return False

    # Insert FITS Keywords in the XML tree
    @staticmethod
    def _insert_fitskeyword(image_xml, keyword_name, keyword_values):
        XISF._validate_fits_keyword_name(keyword_name, writing=True)
        for entry in keyword_values:
            ET.SubElement(
                image_xml,
                "FITSKeyword",
                {
                    "name": keyword_name,
                    "value": XISF._fits_keyword_value(keyword_name, entry),
                    "comment": XISF._fits_keyword_comment(keyword_name, entry),
                },
            )

    # Keywords that have no value of their own, whose value attribute must
    # therefore be an empty string rather than a meaningful value (section 11.6.1)
    _fits_keywords_without_value = ("HISTORY", "COMMENT")

    # The regular expression an Image element id must satisfy (section 11.5.2).
    # Unlike a property identifier, it admits no namespace separator.
    _image_id_re = re.compile(r"[_a-zA-Z][_a-zA-Z0-9]*")

    @staticmethod
    def _validate_fits_keyword_name(name, writing):
        """Check a FITS keyword name against the FITS standard, as quoted in 11.6.1.

        A keyword name shall be a left justified, space-filled ASCII string with
        no embedded spaces, in which all digits 0-9 and the upper case Latin
        letters A-Z are permitted, together with the underscore and the hyphen;
        lower case characters shall not be used. In FITSKeyword elements names
        must not be padded with space characters.

        Args:
            name: the keyword name to check.
            writing: True when encoding, False when decoding. An encoder shall
                generate only conforming units, so an invalid name is an error.
                A decoder must keep the rest of the unit accessible, so an
                invalid name is reported as a warning and sanitized instead.

        Returns:
            The name to use. When writing this is the name unchanged. When
            reading it is a sanitized name that conforms to the above, or the
            original name if it cannot be repaired.
        """
        if not isinstance(name, str):
            name = str(name)
        if XISF._is_valid_fits_keyword_name(name):
            return name

        reason = (
            f"{name!r} is not a valid FITS keyword name: names shall be at most"
            f" 8 characters long, shall not be padded with spaces, and shall"
            f" contain only the digits 0-9, the upper case letters A-Z, the"
            f" underscore and the hyphen (required by the XISF 1.0 spec,"
            f" section 11.6.1)"
        )
        if writing:
            raise XISFError(reason)

        # A decoder may not discard a keyword just because it is malformed, so
        # the name is repaired as far as possible and reported.
        sanitized = name.strip().upper()
        sanitized = "".join(
            c if (c.isascii() and (c.isalnum() or c in "_-")) else "_"
            for c in sanitized
        )
        sanitized = sanitized[:8]
        if sanitized and XISF._is_valid_fits_keyword_name(sanitized):
            warnings.warn(
                f"{reason} Reading it as {sanitized!r}", XISFWarning
            )
            return sanitized
        warnings.warn(
            f"{reason} It cannot be repaired, so the keyword is read with the"
            f" name {name!r} unchanged.",
            XISFWarning,
        )
        return name

    @staticmethod
    def _is_valid_fits_keyword_name(name):
        if len(name) > 8 or not name:
            return False
        return all(
            c.isascii() and (c.isdigit() or ("A" <= c <= "Z") or c in "_-")
            for c in name
        )

    @staticmethod
    def _validate_image_id(image_id):
        """Check an Image element id against the expression quoted in 11.5.2.

        The id attribute is optional, but when it is present image-id shall be a
        sequence of ASCII characters satisfying [_a-zA-Z][_a-zA-Z0-9]*. The
        expression admits neither spaces nor the colon used to delimit
        namespaces in property identifiers, and it requires the first character
        to be a letter or an underscore.

        Unlike a property identifier, there is nothing to repair here: every
        character outside the expression is equally unusable, and any
        substitution would invent an identifier the caller did not ask for. An
        encoder shall generate only conforming units, so this raises.

        Uniqueness within the XISF unit, also required by 11.5.2, is not checked
        here: write() serializes a single Image element, and the reader returns
        the images in document order rather than keyed by id, so a unit written
        by this module cannot violate it.
        """
        if isinstance(image_id, str) and XISF._image_id_re.fullmatch(image_id):
            return
        raise XISFError(
            f"Image id {image_id!r} is not a valid XISF image identifier:"
            f" identifiers shall be ASCII characters matching"
            f" [_a-zA-Z][_a-zA-Z0-9]*, so they shall start with a letter or an"
            f" underscore and shall contain neither spaces nor colons"
            f" (required by the XISF 1.0 spec, section 11.5.2)"
        )

    @staticmethod
    def _fits_keyword_value(keyword_name, entry):
        """Return the value attribute of a FITSKeyword element.

        The value and the comment of a FITSKeyword element are mandatory, but a
        missing one is recoverable: the keyword is still meaningful, so an empty
        string is written in its place and the omission is reported. The
        keywords HISTORY and COMMENT have no value of their own, whose value
        attribute must be an empty string, so a missing value is expected there
        and is not reported.
        """
        value = entry.get("value") if isinstance(entry, dict) else None
        if value is None:
            if keyword_name not in XISF._fits_keywords_without_value:
                XISF._warn_fits_keyword(
                    keyword_name,
                    "value",
                    "the value attribute is mandatory, so an empty string is used",
                )
            return ""
        if not isinstance(value, str):
            return str(value)
        return value

    @staticmethod
    def _fits_keyword_comment(keyword_name, entry):
        """Return the comment attribute of a FITSKeyword element.

        As with the value, a missing mandatory comment is recovered with an
        empty string and reported, since the keyword itself is still useful.
        """
        comment = entry.get("comment") if isinstance(entry, dict) else None
        if comment is None:
            XISF._warn_fits_keyword(
                keyword_name,
                "comment",
                "the comment attribute is mandatory, so an empty string is used",
            )
            return ""
        if not isinstance(comment, str):
            return str(comment)
        return comment

    @staticmethod
    def _warn_fits_keyword(keyword_name, attribute, consequence):
        warnings.warn(
            f"FITSKeyword {keyword_name!r} has no {attribute} attribute:"
            f" {consequence} (the XISF 1.0 spec, section 11.6.1, makes it"
            f" mandatory)",
            XISFWarning,
        )

    # Returns image geometry as a tuple, e.g. (x, y, channels) for a 2-D image
    @staticmethod
    def _parse_geometry(g):
        geometry = tuple(map(int, g.split(":")))
        if len(geometry) < 2:
            # geometry is dim1:...:dimN:channel-count with N >= 1, so a valid
            # geometry has at least one dimension plus the channel count
            raise XISFError(
                f"Invalid geometry {g!r}: expected dim1:...:dimN:channel-count"
                f" with N >= 1 (spec 11.5.1)"
            )
        if 0 in geometry:
            # every dimi item shall be greater than zero (spec 11.5.1)
            raise XISFError(
                f"Invalid geometry {g!r}: every dimension and the channel count"
                f" shall be greater than zero (spec 11.5.1)"
            )
        return geometry

    # Returns ("attachment", position, size), ("inline", encoding) or ("embedded")
    @staticmethod
    def _parse_location(l):
        ll = l.split(":")
        if ll[0] not in ["inline", "embedded", "attachment"]:
            raise NotImplementedError(f"Data block location type '{ll[0]}' not implemented")
        return (ll[0], int(ll[1]), int(ll[2])) if ll[0] == "attachment" else ll

    # Serialize location tuple to string, as value for location attribute
    @staticmethod
    def _to_location(location_tuple):
        return ":".join([str(e) for e in location_tuple])

    # Returns (codec, uncompressed_size, item_size); item_size is None if not using byte shuffling
    @staticmethod
    def _parse_compression(c):
        cl = c.split(":")
        if len(cl) == 3:
            # (codec+byteshuffling, uncompressed_size, shuffling_item_size)
            return (cl[0], int(cl[1]), int(cl[2]))
        else:
            # (codec, uncompressed_size, None)
            return (cl[0], int(cl[1]), None)

    # Return equivalent numpy dtype
    # Cryptographic hashing algorithms of spec 10.5 Table 9, mapping every
    # accepted spelling to the hashlib constructor to use. The first key of each
    # pair is the canonical checksum algorithm value, and the second its
    # alternate. SHA-1, SHA-256 and SHA-512 shall be supported by all decoders,
    # and SHA-1 by all encoders claiming support for checksums.
    _checksum_algorithms = {
        "sha-1": "sha1",
        "sha1": "sha1",
        "sha-256": "sha256",
        "sha256": "sha256",
        "sha-512": "sha512",
        "sha512": "sha512",
        "sha3-256": "sha3_256",
        "sha3-512": "sha3_512",
    }
    # hashlib gained sha3_256 and sha3_512 in Python 3.6, but their presence
    # depends on the OpenSSL build, so they are checked rather than assumed.
    _optional_checksum_algorithms = ("sha3-256", "sha3-512")

    @classmethod
    def _parse_checksum(cls, checksum, owner):
        """Parse a checksum attribute into an (algorithm, digest) pair.

        The syntax is 'algorithm:digest' (spec 10.5), where the digest is
        Base16 encoded with lowercase hexadecimal digits.
        """
        algorithm, sep, digest = checksum.partition(":")
        if not sep or not algorithm or not digest:
            raise XISFError(
                f"{owner} has checksum={checksum!r}: the checksum attribute shall"
                f" have the form 'algorithm:digest' (XISF 1.0 spec, section 10.5)"
            )
        if algorithm not in cls._checksum_algorithms:
            raise XISFError(
                f"{owner} declares unknown checksum algorithm {algorithm!r}: the"
                f" supported algorithms are"
                f" {', '.join(sorted(set(cls._checksum_algorithms)))}"
                f" (XISF 1.0 spec, section 10.5)"
            )
        if algorithm in cls._optional_checksum_algorithms and not hasattr(
            hashlib, cls._checksum_algorithms[algorithm]
        ):
            raise XISFError(
                f"{owner} declares checksum algorithm {algorithm!r}, which this"
                f" Python build does not provide"
            )
        return algorithm, digest.lower()

    @staticmethod
    def _validate_checksum_algorithm(algorithm):
        """Normalize the encoder's checksum argument to a canonical algorithm name.

        Accepts True as the spec-recommended SHA-1, and False or None to write no
        checksums at all. The alternate spellings of Table 9 are accepted and
        normalized to their canonical value.
        """
        if algorithm is None or algorithm is False:
            return None
        if algorithm is True:
            # SHA-1 is the recommended general-purpose algorithm (spec 10.5)
            return "sha-1"
        if not isinstance(algorithm, str):
            raise XISFError(
                f"checksum must be a checksum algorithm name, True or False, got"
                f" {type(algorithm).__name__}"
            )
        for canonical, aliases in (
            ("sha-1", ("sha-1", "sha1")),
            ("sha-256", ("sha-256", "sha256")),
            ("sha-512", ("sha-512", "sha512")),
            ("sha3-256", ("sha3-256",)),
            ("sha3-512", ("sha3-512",)),
        ):
            if algorithm in aliases:
                return canonical
        raise XISFError(
            f"unknown checksum algorithm {algorithm!r}: the supported algorithms"
            f" are sha-1, sha-256, sha-512, sha3-256 and sha3-512 (XISF 1.0 spec,"
            f" section 10.5)"
        )

    @classmethod
    def _compute_checksum(cls, algorithm, data):
        """Compute the Base16 digest of a serialized data block.

        The digest is taken over the block as it is serialized, which for inline
        and embedded blocks is the decoded binary data rather than its Base64 or
        Base16 text (spec 10.5), and for a compressed block is the compressed data
        (spec 10.6.1). Message digests are Base16 encoded with lowercase
        hexadecimal digits.
        """
        digest = hashlib.new(cls._checksum_algorithms[algorithm], data).hexdigest()
        return digest.lower()

    @staticmethod
    def _dtype_for_byte_order(dtype, byte_order):
        """Return the dtype to interpret a serialized block with.

        The byteOrder attribute declares the endianness of a data block, and
        little-endian is assumed when it is absent (spec 10.4). A baseline
        decoder shall read data blocks in both byte orders (spec 7.2).

        The returned dtype is the one that correctly interprets the serialized
        bytes, which is the explicit byte order of the block. It differs from
        the native-order dtype exactly when the block has to be byte-swapped,
        which read_image then does so the caller always gets a native-order
        array. Endianness is immaterial for single-byte sample formats, where
        the block order and the native order are the same dtype.
        """
        if byte_order not in ("little", "big"):
            raise XISFError(
                f"Image declares byteOrder={byte_order!r}: the byteOrder"
                f" attribute must be either 'big' or 'little' (XISF 1.0 spec,"
                f" section 10.4)"
            )
        if byte_order == "little" or dtype.itemsize == 1:
            return dtype
        return dtype.newbyteorder(">" if byte_order == "big" else "<")

    @staticmethod
    def _parse_sampleFormat(s):
        # Translate alternate names to "canonical" type names
        alternate_names = {
            'Byte': 'UInt8',
            'Short': 'Int16',
            'UShort': 'UInt16',
            'Int': 'Int32',
            'UInt': 'UInt32',
            'Float': 'Float32',
            'Double': 'Float64',
        }
        try:
            s = alternate_names[s]
        except KeyError:
            pass

        _dtypes = {
            "UInt8": np.dtype("uint8"),
            "UInt16": np.dtype("uint16"),
            "UInt32": np.dtype("uint32"),
            "Float32": np.dtype("float32"),
            "Float64": np.dtype("float64"),
        }
        try:
            return _dtypes[s]
        except:
            raise NotImplementedError(f"sampleFormat {s} not implemented")

    # Return XISF data type from numpy dtype
    @staticmethod
    def _get_sampleFormat(dtype):
        _sampleFormats = {
            "uint8": "UInt8",
            "uint16": "UInt16",
            "uint32": "UInt32",
            "float32": "Float32",
            "float64": "Float64",
        }
        try:
            return _sampleFormats[dtype.name]
        except:
            raise NotImplementedError(f"sampleFormat for {dtype} not implemented")

    @staticmethod
    def _parse_vector_dtype(type_name):
        # Translate alternate names to "canonical" type names
        alternate_names = {
            'ByteArray': 'UI8Vector',
            'IVector': 'I32Vector',
            'UIVector': 'UI32Vector',
            'Vector': 'F64Vector',
        }
        try:
            type_name = alternate_names[type_name]
        except KeyError:
            pass

        type_prefix = type_name[:-6]  # removes "Vector" and "Matrix" suffixes
        _dtypes = {
            "I8": np.dtype("int8"),
            "UI8": np.dtype("uint8"),
            "I16": np.dtype("int16"),
            "UI16": np.dtype("uint16"),
            "I32": np.dtype("int32"),
            "UI32": np.dtype("uint32"),
            "I64": np.dtype("int64"),
            "UI64": np.dtype("uint64"),
            "F32": np.dtype("float32"),
            "F64": np.dtype("float64"),
            "C32": np.dtype("csingle"),
            "C64": np.dtype("cdouble"),
        }
        try:
            return _dtypes[type_prefix]
        except:
            raise NotImplementedError(f"data type {type_name} not implemented")

    # __/ Auxiliary functions for compression/shuffling \________

    # Un-byteshuffling implementation based on numpy
    @staticmethod
    def _unshuffle(d, item_size):
        a = np.frombuffer(d, dtype=np.dtype("uint8"))
        a = a.reshape((item_size, -1))
        return np.transpose(a).tobytes()

    # Byteshuffling implementation based on numpy
    @staticmethod
    def _shuffle(d, item_size):
        a = np.frombuffer(d, dtype=np.dtype("uint8"))
        a = a.reshape((-1, item_size))
        return np.transpose(a).tobytes()

    # LZ4/zlib/zstd decompression
    @staticmethod
    def _decompress(data, elem):
        # (codec, uncompressed-size, item-size); item-size is None if not using byte shuffling
        codec, uncompressed_size, item_size = elem["compression"]

        if codec.startswith("lz4"):
            data = lz4.block.decompress(data, uncompressed_size=uncompressed_size)
        elif codec.startswith("zstd"):
            data = zstandard.decompress(data, max_output_size=uncompressed_size)
        elif codec.startswith("zlib"):
            data = zlib.decompress(data)
        else:
            raise NotImplementedError(f"Unimplemented compression codec {codec}")

        if item_size:  # using byte-shuffling
            data = XISF._unshuffle(data, item_size)

        return data

    # LZ4/zlib/zstd compression
    @staticmethod
    def _compress(data, codec, level=None, shuffle=False, itemsize=None):
        compressed = XISF._shuffle(data, itemsize) if shuffle else data

        if codec == "lz4hc":
            level = level if level else XISF._compression_def_level["lz4hc"]
            compressed = lz4.block.compress(
                compressed, mode="high_compression", compression=level, store_size=False
            )
        elif codec == "lz4":
            compressed = lz4.block.compress(compressed, store_size=False)
        elif codec == "zstd":
            level = level if level else XISF._compression_def_level["zstd"]
            compressed = zstandard.compress(compressed, level=level)
        elif codec == "zlib":
            level = level if level else XISF._compression_def_level["zlib"]
            compressed = zlib.compress(compressed, level=level)
        else:
            raise NotImplementedError(f"Unimplemented compression codec {codec}")

        return compressed
