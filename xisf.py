# coding: utf-8

"""
Very crude and incomplete XISF Decoder (see https://pixinsight.com/xisf/).

This implementation is not endorsed nor related with PixInsight development team.

Copyright (C) 2021 Sergio Díaz, sergiodiaz.eu

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

import xml.etree.ElementTree as ET
import numpy as np
import lz4.block # https://python-lz4.readthedocs.io/en/stable/lz4.block.html
import zlib # https://docs.python.org/3/library/zlib.html


# Auxiliary functions to parse some metadata attributes
# Returns image shape, e.g. (x, y, channels)
def _parse_geometry(g):
    return tuple(map(int, g.split(':')))

# Returns (position, size)
def _parse_location(l):
    ll = l.split(':')
    if ll[0] != 'attachment':
        raise NotImplementedError("Image location type '%s' not implemented" % (ll[0],))
    return tuple(map(int, ll[1:]))

# Return equivalent numpy dtype
def _parse_sampleFormat(s):
    _dtypes = { 
        'UInt8': np.dtype('uint8'),
        'UInt16': np.dtype('uint16'),
        'UInt32': np.dtype('uint32'),
        'Float32': np.dtype('float32'),
        'Float64': np.dtype('float64'),
    }
    try:
        dtype = _dtypes[s]            
    except:
        raise NotImplementedError("sampleFormat %s not implemented" % (s,))
    return dtype

# Returns (codec, uncompressed_size, item_size); item_size is None if not using byte shuffling
def _parse_compression(c):       
    cl = c.split(':')
    if len(cl) == 3: # (codec+byteshuffling, uncompressed_size, shuffling_item_size)
        return (cl[0], int(cl[1]), int(cl[2]))
    else:  # (codec, uncompressed_size, None)
        return (cl[0], int(cl[1]), None)

# Auxiliary function to implement un-byteshuffling with numpy
def _unshuffle(d, item_size):
    a = np.frombuffer(d, dtype=np.dtype('uint8'))
    a = a.reshape((item_size, -1))
    return np.transpose(a).tobytes()


class XISF:
    """Implements an uncomplete XISF Decoder. It parses file and attached images metadata. Image data is returned as a 
    numpy ndarray, using the "channels last" convention. 

    What's supported: 
     - Reads Monolithic XISF files
     - Read multiple Image core elements from a monolithic XISF file
     - Support all standard compression codecs defined in this specification for decompression (zlib/lz4[hc]+
     byte shuffling)
     - Read pixel data from XISF blocks with attachment block locations
     - Read pixel data in the planar pixel storage models, *however it assumes 2D images only* (with multiple channels)
     - Read pixel data in the UInt8/16/32 and Float32/64 pixel sample formats
     - Read pixel data encoded in the grayscale and RGB color spaces
     - "Atomic" properties only (Scalar, Strings, TimePoint)

    What's not supported (at least by now):
     - Read pixel data from XISF blocks with inline or embedded block locations
     - Read pixel data in the normal pixel storage models
     - Read pixel data in the planar pixel storage models other than 2D images
     - Complex, Vector, Matrix and Table properties

    Usage example:
    >>> import XISF
    >>> import matplotlib.pyplot as plt
    >>> xisf = XISF()
    >>> xisf.read("file.xisf")
    >>> ims_meta = xisf.get_images_metadata()
    >>> ims_meta
    >>> im_data = xisf.read_image(0)
    >>> plt.imshow(im_data)
    >>> xisf.close()

    The XISF format specification is available at https://pixinsight.com/doc/docs/XISF-1.0-spec/XISF-1.0-spec.html
    """

    _signature = b'XISF0100' # Monolithic
    _headerlength_len = 4
    _reserved_len = 4
    _xml_ns = { 'xisf': "http://www.pixinsight.com/xisf" }


    def __init__(self):
        self._f = None
        self._headerlength = None
        self._xisf_header = None
        self._xisf_header_xml = None
        self._images_meta = None
        self._file_meta = None
        ET.register_namespace('', "http://www.pixinsight.com/xisf")


    def read(self, fname):
        """Opens a XISF file and extract its metadata. It is mandatory before any other operation.

        Args:
            fname: filename
        
        Returns:
            (Nothing)

        """
        self._f = open(fname, "rb")

        # Check XISF signature
        signature = self._f.read(len(self._signature))
        if signature != self._signature:
            raise ValueError("File doesn't have XISF signature")

        # Get header length
        self._headerlength = int.from_bytes(
            self._f.read(self._headerlength_len),
            byteorder='little'
        )
        # Equivalent:
        # self._headerlength = np.fromfile(self._f, dtype=np.uint32, count=1)[0]

        # Skip reserved field
        _ = self._f.read(self._reserved_len)

        # Get XISF (XML) Header
        self._xisf_header = self._f.read(self._headerlength)
        self._xisf_header_xml = ET.fromstring(self._xisf_header)

        # Analyze header to get Data Blocks position and length
        self._images_meta = []
        for image in self._xisf_header_xml.findall('xisf:Image', self._xml_ns):
            image_basic_meta = image.attrib
            # Parse and replace geometry and location with tuples, 
            # parses and translates sampleFormat to numpy dtypes, 
            # and extend with metadata from children entities (FITSKeywords, XISFProperties)
            image_extended_meta = {
                'geometry': _parse_geometry(image.attrib['geometry']),
                'location': _parse_location(image.attrib['location']), 
                'dtype': _parse_sampleFormat(image.attrib['sampleFormat']), 
                'FITSKeywords': {a.attrib['name']: a.attrib['value'] 
                    for a in image.findall('xisf:FITSKeyword', self._xml_ns)
                },
                'XISFProperties': {a.attrib['id']: a.attrib.get('value', a.text) 
                    for a in image.findall('xisf:Property', self._xml_ns)
                }
            }
            # Also parses compression attribute if present, converting it to a tuple
            if 'compression' in image.attrib:
                image_extended_meta['compression'] = _parse_compression(image.attrib['compression'])

            # Merge basic and extended metadata in a dict 
            image_meta = {**image_basic_meta, **image_extended_meta}

            # Append the image metadata to the list
            self._images_meta.append(image_meta)

        # Analyze header for file metadata
        self._file_meta = {}
        for p in self._xisf_header_xml.find('xisf:Metadata', self._xml_ns):
            self._file_meta[p.attrib['id']] = p.attrib.get('value', p.text)


    def get_images_metadata(self):
        """Provides the metadata of all image blocks contained in the XISF File, extracted from 
        the header (<Image> tags). To get the actual image data, see read_image().
        
        Returns:
            list [ m_0, m_1, ..., m_{n-1} ] where m_i is a dict with the metadata of the image i:
            m_i = { 
                'geometry': (width, height, channels), # only 2D images (with multiple channels) are supported
                'location': (pos, size), # used internally in read_image()
                'dtype': np.dtype('...'), # derived from sampleFormat argument
                'compression': (codec, uncompressed_size, item_size), # optional
                'key': 'value', # other <Image> attributes are simply copied 
                ..., 
                'FITSKeywords': { 'key': 'value', ... }, # child attributes are also copied
                'XISFProperties': { 'key': 'value', ... }
            }

        """          
        return self._images_meta


    def get_file_metadata(self):
        """Provides the metadata from the header of the XISF File (<Metadata> tag).
        
        Returns:
            dict with the properties of the metadata as key-value pairs.

        """          
        return self._file_meta


    def read_image(self, n):
        """Extracts an image from a XISF file already opened with read().

        Args:
            n: index of the image to extract in the list returned by get_images_metadata()
        
        Returns:
            Numpy ndarray with the image data, in "channels-last" format.

        """        
        try:
            meta = self._images_meta[n]
        except IndexError as e:
            if self._xisf_header is None:
                raise RuntimeError("No file loaded") from e
            elif not self._images_meta:
                raise ValueError("File does not contain image data") from e
            else:
                raise ValueError("Requested image #%d, valid range is [0..%d]" % (n, len(self._images_meta)-1)) from e
        
        pos, size = meta['location'] # Position and size of the Data Block containing the image data

        try:
            w, h, chc = meta['geometry'] # Assumes *two*-dimensional images (chc=channel count)
        except ValueError as e:
            raise NotImplementedError("Assumed 2D channels (width, height, channels), found %s geometry" % (meta['geometry'],))

        if 'compression' in meta:
            # (codec, uncompressed-size, item-size); item-size is None if not using byte shuffling
            codec, uncompressed_size, item_size = meta['compression'] 
            self._f.seek(pos)
            im_data = self._f.read(size)

            if codec.startswith("lz4"):
                im_data = lz4.block.decompress(im_data, uncompressed_size=uncompressed_size)
            elif codec.startswith("zlib"):
                im_data = zlib.decompress(im_data)
            else:
                raise NotImplementedError("Unimplemented compression codec %s" % (codec,))

            if item_size: # using byte-shuffling
                im_data = _unshuffle(im_data, item_size)
            
            im_data = np.frombuffer(im_data, dtype=meta['dtype'])

        else:
            # no compression
            self._f.seek(0)
            im_data = np.fromfile(self._f, offset=pos, dtype=meta['dtype'], count=h*w*chc)

        im_data = im_data.reshape((chc,h,w))
        return np.transpose(im_data, (1, 2, 0)) # channels-last convention (tensorflow, matplotlib's imshow, etc)


    def close(self):
        self._f.close()
        