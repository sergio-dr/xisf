# coding: utf-8

"""
(Incomplete) XISF Encoder/Decoder (see https://pixinsight.com/xisf/).

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
import sys
from datetime import datetime


class XISF:
    """Implements an *incomplete* (attached images only) baseline XISF Decoder and a simple baseline Encoder. 
    It parses metadata from Image and Metadata XISF core elements. Image data is returned as a numpy ndarray 
    (using the "channels-last" convention by default). 

    What's supported: 
    - Monolithic XISF files only
        - XISF blocks with attachment block locations (neither inline nor embedded block locations as required 
          for a complete baseline decoder)
        - Planar pixel storage models, *however it assumes 2D images only* (with multiple channels)
        - UInt8/16/32 and Float32/64 pixel sample formats
        - Grayscale and RGB color spaces     
    - Decoding:
        - multiple Image core elements from a monolithic XISF file
        - Support all standard compression codecs defined in this specification for decompression (zlib/lz4[hc]+
          byte shuffling)
    - Encoding:
        - Single image core element
        - Uncompressed data blocks only       
    - "Atomic" properties only (Scalar, Strings, TimePoint)
    - Metadata and FITSKeyword core elements

    What's not supported (at least by now):
    - Read pixel data from XISF blocks with inline or embedded block locations
    - Read pixel data in the normal pixel storage models
    - Read pixel data in the planar pixel storage models other than 2D images
    - Complex, Vector, Matrix and Table properties
    - Any other not explicitly supported core elements (Resolution, Thumbnail, ICCProfile, etc.)

    Usage example:
    ```
    >>> from xisf import XISF
    >>> import matplotlib.pyplot as plt
    >>> xisf = XISF("file.xisf")
    >>> file_meta = xisf.get_file_metadata()    
    >>> file_meta
    >>> ims_meta = xisf.get_images_metadata()
    >>> ims_meta
    >>> im_data = xisf.read_image(0)
    >>> plt.imshow(im_data)
    >>> plt.show()
    >>> XISF.write("output.xisf", im_data, ims_meta[0], file_meta)
    ```

    If the file is not huge and it contains only an image (or you're interested just in one of the 
    images inside the file), there is a convenience method for reading the data and the metadata:
    ```
    >>> from xisf import XISF
    >>> import matplotlib.pyplot as plt    
    >>> im_data = XISF.read("file.xisf")
    >>> plt.imshow(im_data)
    >>> plt.show()
    ```

    The XISF format specification is available at https://pixinsight.com/doc/docs/XISF-1.0-spec/XISF-1.0-spec.html
    """

    # Static attributes
    _signature = b'XISF0100' # Monolithic
    _headerlength_len = 4
    _reserved_len = 4
    _property_types = {
        "XISF:CreationTime": "TimePoint",
        "XISF:CreatorApplication": "String",
        "XISF:Abstract": "String",
        "XISF:AccessRights": "String",
        "XISF:Authors": "String",
        "XISF:BibliographicReferences": "String",
        "XISF:BriefDescription": "String",
        "XISF:CompressionLevel": "Int32",
        "XISF:CompressionCodecs": "String",
        "XISF:Contributors": "String",
        "XISF:Copyright": "String",
        "XISF:CreatorModule": "String",
        "XISF:CreatorOS": "String",
        "XISF:Description": "String",
        "XISF:Keywords": "String",
        "XISF:Languages": "String",
        "XISF:License": "String",
        "XISF:OriginalCreationTime": "TimePoint",
        "XISF:RelatedResources": "String",
        "XISF:Title": "String",
        "Observer:EmailAddress": "String",
        "Observer:Name": "String",
        "Observer:PostalAddress": "String",
        "Observer:Website": "String",
        "Organization:EmailAddress": "String",
        "Organization:Name": "String",
        "Organization:PostalAddress": "String",
        "Organization:Website": "String",
        "Observation:CelestialReferenceSystem": "String",
        "Observation:BibliographicReferences": "String",
        "Observation:Center:Dec": "Float64",
        "Observation:Center:RA": "Float64",
        "Observation:Center:X": "Float64",
        "Observation:Center:Y": "Float64",
        "Observation:Description": "String",
        "Observation:Equinox": "Float64",
        "Observation:GeodeticReferenceSystem": "String",
        "Observation:Location:Elevation": "Float64",
        "Observation:Location:Latitude": "Float64",
        "Observation:Location:Longitude": "Float64",
        "Observation:Location:Name": "String",
        "Observation:Meteorology:AmbientTemperature": "Float32",
        "Observation:Meteorology:AtmosphericPressure": "Float32",
        "Observation:Meteorology:RelativeHumidity": "Float32",
        "Observation:Meteorology:WindDirection": "Float32",
        "Observation:Meteorology:WindGust": "Float32",
        "Observation:Meteorology:WindSpeed": "Float32",
        "Observation:Object:Dec": "Float64",
        "Observation:Object:Name": "String",
        "Observation:Object:RA": "Float64",
        "Observation:RelatedResources": "String",
        "Observation:Time:End": "TimePoint",
        "Observation:Time:Start": "TimePoint",
        "Observation:Title": "String",
        "Instrument:Camera:Gain": "Float32",
        "Instrument:Camera:ISOSpeed": "Int32",
        "Instrument:Camera:Name": "String",
        "Instrument:Camera:ReadoutNoise": "Float32",
        "Instrument:Camera:Rotation": "Float32",
        "Instrument:Camera:XBinning": "Int32",
        "Instrument:Camera:YBinning": "Int32",
        "Instrument:ExposureTime": "Float32",
        "Instrument:Filter:Name": "String",
        "Instrument:Focuser:Position": "Float32",
        "Instrument:Sensor:TargetTemperature": "Float32",
        "Instrument:Sensor:Temperature": "Float32",
        "Instrument:Sensor:XPixelSize": "Float32",
        "Instrument:Sensor:YPixelSize": "Float32",
        "Instrument:Telescope:Aperture": "Float32",
        "Instrument:Telescope:CollectingArea": "Float32",
        "Instrument:Telescope:FocalLength": "Float32",
        "Instrument:Telescope:Name": "String",
        "Image:FrameNumber": "UInt32",
        "Image:GroupId": "String",
        "Image:SubgroupId": "String",
        "Processing:Description": "String",
        "Processing:History": "String"
    }
    _xml_ns = { 'xisf': "http://www.pixinsight.com/xisf" }    
    _xisf_attrs = {
        'xmlns': "http://www.pixinsight.com/xisf",
        'xmlns:xsi': "http://www.w3.org/2001/XMLSchema-instance",
        'version': "1.0",
        'xsi:schemaLocation': "http://www.pixinsight.com/xisf http://pixinsight.com/xisf/xisf-1.0.xsd"
    }


    def __init__(self, fname):
        """Opens a XISF file and extract its metadata. To get the metadata and the images, see get_file_metadata(), 
        get_images_metadata() and read_image().
        Args:
            fname: filename
        
        Returns:
            XISF object.
        """        
        self._fname = fname
        self._headerlength = None
        self._xisf_header = None
        self._xisf_header_xml = None
        self._images_meta = None
        self._file_meta = None
        ET.register_namespace('', self._xml_ns['xisf'])

        self._read()


    def _read(self):
        with open(self._fname, "rb") as f:
            # Check XISF signature
            signature = f.read(len(self._signature))
            if signature != self._signature:
                raise ValueError("File doesn't have XISF signature")

            # Get header length
            self._headerlength = int.from_bytes(
                f.read(self._headerlength_len),
                byteorder = 'little'
            )
            # Equivalent:
            # self._headerlength = np.fromfile(f, dtype=np.uint32, count=1)[0]

            # Skip reserved field
            _ = f.read(self._reserved_len)

            # Get XISF (XML) Header
            self._xisf_header = f.read(self._headerlength)
            self._xisf_header_xml = ET.fromstring(self._xisf_header)
        self._analyze_header()

    def _analyze_header(self):
        # Analyze header to get Data Blocks position and length
        self._images_meta = []
        for image in self._xisf_header_xml.findall('xisf:Image', self._xml_ns):
            image_basic_meta = image.attrib
            # Parse and replace geometry and location with tuples, 
            # parses and translates sampleFormat to numpy dtypes, 
            # and extend with metadata from children entities (FITSKeywords, XISFProperties)
            # TODO: Resolution, ICCProfile, Thumbnail, ...
            
            # The same FITS keyword can appear multiple times, so we have to 
            # prepare a dict of lists. Each element in the list is a dict
            # that hold the value and the comment associated with the keyword.
            # Not as clear as I would like. 
            fits_keywords = {}
            for a in image.findall('xisf:FITSKeyword', self._xml_ns):
                fits_keywords.setdefault(a.attrib['name'], []).append({
                    'value': a.attrib['value'].strip("'").strip(" "),
                    'comment': a.attrib['comment'],
                })
            image_extended_meta = {
                'geometry': self._parse_geometry(image.attrib['geometry']),
                'location': self._parse_location(image.attrib['location']), 
                'dtype': self._parse_sampleFormat(image.attrib['sampleFormat']), 
                'FITSKeywords': fits_keywords,
                'XISFProperties': {a.attrib['id']: a.attrib.get('value', a.text) 
                    for a in image.findall('xisf:Property', self._xml_ns)
                }
            }
            # Also parses compression attribute if present, converting it to a tuple
            if 'compression' in image.attrib:
                image_extended_meta['compression'] = self._parse_compression(image.attrib['compression'])

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
        the header (<Image> core elements). To get the actual image data, see read_image().

        It outputs a dictionary m_i for each image, with the following structure:
        ```
        m_i = { 
            'geometry': (width, height, channels), # only 2D images (with multiple channels) are supported
            'location': (pos, size), # used internally in read_image()
            'dtype': np.dtype('...'), # derived from sampleFormat argument
            'compression': (codec, uncompressed_size, item_size), # optional
            'key': 'value', # other <Image> attributes are simply copied 
            ..., 
            'FITSKeywords': { <fits_keyword>: [ {'value': <value>, 'comment': <comment> }, ...], ... }, 
            'XISFProperties': { <xisf_property_name>: value, ... }
        }
        ```

        Returns:
            list [ m_0, m_1, ..., m_{n-1} ] where m_i is a dict as described above.
 
        """          
        return self._images_meta


    def get_file_metadata(self):
        """Provides the metadata from the header of the XISF File (<Metadata> core elements).
        
        Returns:
            dict with the properties of the metadata as key-value pairs.

        """          
        return self._file_meta


    def read_image(self, n=0, data_format='channels_last'):
        """Extracts an image from a XISF object.

        Args:
            n: index of the image to extract in the list returned by get_images_metadata()
            data_format: channels axis can be 'channels_first' or 'channels_last' (as used in 
            keras/tensorflow, pyplot's imshow, etc.), 0 by default.
        
        Returns:
            Numpy ndarray with the image data, in the requested format (channels_first or channels_last).

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

        with open(self._fname, "rb") as f:
            if 'compression' in meta:
                # (codec, uncompressed-size, item-size); item-size is None if not using byte shuffling
                codec, uncompressed_size, item_size = meta['compression'] 
                f.seek(pos)
                im_data = f.read(size)

                if codec.startswith("lz4"):
                    im_data = lz4.block.decompress(im_data, uncompressed_size=uncompressed_size)
                elif codec.startswith("zlib"):
                    im_data = zlib.decompress(im_data)
                else:
                    raise NotImplementedError("Unimplemented compression codec %s" % (codec,))

                if item_size: # using byte-shuffling
                    im_data = self._unshuffle(im_data, item_size)
                
                im_data = np.frombuffer(im_data, dtype=meta['dtype'])

            else:
                # no compression
                f.seek(0)
                im_data = np.fromfile(f, offset=pos, dtype=meta['dtype'], count=h*w*chc)

        im_data = im_data.reshape((chc,h,w))
        return np.transpose(im_data, (1, 2, 0)) if data_format == 'channels_last' else im_data


    @staticmethod
    def read(fname, n=0, image_metadata={}, xisf_metadata={}):
        """Convenience method for reading a file containing a single image. 

        Args:
            fname (string): filename
            n (int, optional): index of the image to extract (in the list returned by get_images_metadata()). Defaults to 0.
            image_metadata (dict, optional): dictionary that will be updated with the metadata of the image.
            xisf_metadata (dict, optional): dictionary that will be updated with the metadata of the file.

        Returns:
            [np.ndarray]: Numpy ndarray with the image data, in the requested format (channels_first or channels_last).
        """
        xisf = XISF(fname)
        xisf_metadata.update( xisf.get_file_metadata() )
        image_metadata.update( xisf.get_images_metadata()[n] )
        return xisf.read_image(n)


    # if 'colorSpace' is not specified, im_data.shape[2] dictates if colorSpace is 'Gray' or 'RGB' 
    # For float sample formats, bounds="0:1" is assumed
    @staticmethod
    def write(fname, im_data, image_metadata={}, xisf_metadata={}):
        """Writes an image (numpy array) to a XISF file.

        Args:
            fname: filename (will overwrite if existing)
            im_data: numpy ndarray with the image data
            image_metadata: dict with the same structure described for m_i in get_images_metadata(). 
              Only 'FITSKeywords' and 'XISFProperties' keys are actually written, the rest are derived from im_data.
            xisf_metadata: dict with the same structure returned by get_file_metadata()
        
        Returns:
            Nothing

        """          
        # Update Image metadata
        image_attrs = {}
        if im_data.shape[2] == 3 or  im_data.shape[2] == 1:
            data_format = 'channels_last'
            geometry = (im_data.shape[1], im_data.shape[0], im_data.shape[2])
            channels = im_data.shape[2]
        else:
            data_format = 'channels_first'
            geometry = im_data.shape
            channels = im_data.shape[0]
        image_attrs['geometry'] = "%d:%d:%d" % geometry
        uncompressed_size = str(im_data.size * im_data.itemsize) # TODO compression size
        image_attrs['location'] = ':'.join( ('attachment', "", uncompressed_size) ) # provisional until we get the data block position
        image_attrs['colorSpace'] = image_attrs.get('colorSpace', 'Gray' if channels == 1 else 'RGB')
        image_attrs['sampleFormat'] = XISF._get_sampleFormat(im_data.dtype)
        if image_attrs['sampleFormat'].startswith("Float"):
            image_attrs['bounds'] = "0:1" # Assumed
        if sys.byteorder == 'big' and image_attrs['sampleFormat'] != 'UInt8':
            image_attrs['byteOrder'] = 'big'
        # TODO: compression

        # Create file metadata
        xisf_metadata['XISF:CreationTime'] = datetime.utcnow().isoformat()
        xisf_metadata['XISF:CreatorApplication'] = "Python"
        xisf_metadata['XISF:CreatorModule'] = "XISF Python Module"
        _OSes = {
            'linux': 'Linux',
            'win32': 'Windows',
            'cygwin': 'Windows',
            'darwin': 'macOS'
        }
        xisf_metadata['XISF:CreatorOS'] = _OSes[sys.platform]
        # TODO: compression


        # Convert metadata (dict) to XML Header
        xisf_header_xml = ET.Element('xisf', XISF._xisf_attrs)

        image_xml = ET.SubElement(xisf_header_xml, 'Image', image_attrs)
        # XISFProperties
        for property_id, value in image_metadata.get('XISFProperties', {}).items():
            try:
                property_type = XISF._property_types[property_id] # TODO: error handling

                if property_type == 'String':
                    ET.SubElement(image_xml, 'Property', {
                        'id': property_id,
                        'type': property_type
                    }).text = value
                else:        
                    ET.SubElement(image_xml, 'Property', {
                        'id': property_id,
                        'type': property_type, 
                        'value': value
                    })                
            except KeyError as e:
                print("Warning: unknown Image property %s" % (property_id,))

        # FITSKeywords
        for keyword_name, data in image_metadata.get('FITSKeywords', {}).items():
            for entry in data:
                ET.SubElement(image_xml, 'FITSKeyword', {
                    'name': keyword_name,
                    'value': entry['value'],
                    'comment': entry['comment']
                })


        metadata_xml = ET.SubElement(xisf_header_xml, 'Metadata')
        for property_id, value in xisf_metadata.items():
            try:
                property_type = XISF._property_types[property_id] # TODO: error handling
            except KeyError as e:
                print("Warning: unknown Metadata property %s" % (property_id,))

            if property_type == 'String':
                ET.SubElement(metadata_xml, 'Property', {
                    'id': property_id,
                    'type': property_type,
                }).text = value
            else:
                ET.SubElement(metadata_xml, 'Property', {
                    'id': property_id,
                    'type': property_type, 
                    'value': value
                })

        # Headers combined length without attachment position in XML header
        provisional_xisf_header = ET.tostring(xisf_header_xml, encoding='utf8')
        len_wo_pos = len(XISF._signature) + XISF._headerlength_len + len(provisional_xisf_header) + XISF._reserved_len
        # First estimation of data block position
        provisional_pos = len_wo_pos + len(str(len_wo_pos))
        # Definitive data block position
        data_block_pos = len_wo_pos + len(str(provisional_pos))
        # Update data block position in XML Header
        image_attrs['location'] = ':'.join( ('attachment', str(data_block_pos), uncompressed_size) ) # TODO: compressed size
        image_xml.set('location', image_attrs['location'])

        with open(fname, "wb") as f:
            # Write XISF signature
            f.write(XISF._signature)

            xisf_header = ET.tostring(xisf_header_xml, encoding='utf8')
            headerlength = len(xisf_header)
            # Write header length
            f.write(headerlength.to_bytes(XISF._headerlength_len, byteorder='little'))

            # Write reserved field
            reserved_field = (0).to_bytes(XISF._reserved_len, byteorder='little')
            f.write(reserved_field)

            # Write header
            f.write(xisf_header)

            # Write data block
            assert(data_block_pos == f.tell())
            data_block = np.transpose(im_data, (2, 0, 1)) if data_format == 'channels_last' else im_data
            data_block.tofile(f)


    # Auxiliary functions to parse some metadata attributes
    # Returns image shape, e.g. (x, y, channels)
    @staticmethod
    def _parse_geometry(g):
        return tuple(map(int, g.split(':')))


    # Returns (position, size)
    @staticmethod
    def _parse_location(l):
        ll = l.split(':')
        if ll[0] != 'attachment':
            raise NotImplementedError("Image location type '%s' not implemented" % (ll[0],))
        return tuple(map(int, ll[1:]))


    # Return equivalent numpy dtype
    @staticmethod
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


    # Return XISF data type from numpy dtype
    @staticmethod
    def _get_sampleFormat(dtype):
        _sampleFormats = { 
            'uint8': 'UInt8',
            'uint16': 'UInt16',
            'uint32': 'UInt32',
            'float32': 'Float32',
            'float64': 'Float64',
        }
        try:
            sampleFormat = _sampleFormats[str(dtype)]
        except:
            raise NotImplementedError("sampleFormat %s not implemented" % (dtype,))
        return sampleFormat


    # Returns (codec, uncompressed_size, item_size); item_size is None if not using byte shuffling
    @staticmethod
    def _parse_compression(c):       
        cl = c.split(':')
        if len(cl) == 3: # (codec+byteshuffling, uncompressed_size, shuffling_item_size)
            return (cl[0], int(cl[1]), int(cl[2]))
        else:  # (codec, uncompressed_size, None)
            return (cl[0], int(cl[1]), None)


    # Auxiliary function to implement un-byteshuffling with numpy
    @staticmethod
    def _unshuffle(d, item_size):
        a = np.frombuffer(d, dtype=np.dtype('uint8'))
        a = a.reshape((item_size, -1))
        return np.transpose(a).tobytes()        