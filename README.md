<a id="xisf"></a>

# xisf

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

<a id="xisf.XISF"></a>

## XISF Objects

```python
class XISF()
```

Implements an baseline XISF Decoder and a simple baseline Encoder.
It parses metadata from Image and Metadata XISF core elements. Image data is returned as a numpy ndarray 
(using the "channels-last" convention by default). 

What's supported: 
- Monolithic XISF files only
    - XISF data blocks with attachment, inline or embedded block locations 
    - Planar pixel storage models, *however it assumes 2D images only* (with multiple channels)
    - UInt8/16/32 and Float32/64 pixel sample formats
    - Grayscale and RGB color spaces     
- Decoding:
    - multiple Image core elements from a monolithic XISF file
    - Support all standard compression codecs defined in this specification for decompression 
      (zlib/lz4[hc]/zstd + byte shuffling)
    - Verification of the SHA-1, SHA-256 and SHA-512 checksums that a baseline decoder shall support (spec 7.2), and of the optional SHA3-256 and SHA3-512 ones. A compressed block is verified before it is decompressed (spec 10.6.1).
- Encoding:
    - Single image core element with an attached data block
    - Support all standard compression codecs defined in this specification for decompression 
      (zlib/lz4[hc]/zstd + byte shuffling)
    - SHA-1 checksums of every data block by default, with any of the five algorithms of spec 10.5 available (can be disabled)
- "Atomic" properties (scalar types, String, TimePoint), Vector and Matrix (e.g. astrometric solutions)
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

<a id="xisf.XISF.__init__"></a>

#### \_\_init\_\_

```python
def __init__(fname)
```

Opens a XISF file and extract its metadata. To get the metadata and the images, see get_file_metadata(),
get_images_metadata() and read_image().

**Arguments**:

- `fname` - filename
- `verify_checksums` - whether to verify the checksum attribute of data blocks that declare one (spec 10.5).
  Verification happens when the block is read, not when the file is opened, and covers image and property data
  blocks alike. When a digest does not match, the block is not made available and XISFError is raised; set to
  False to downgrade the failure to an XISFWarning and return the data anyway. Passing False does not disable
  the check that a compressed block is never decompressed after a failed verification (spec 10.6.1).
  

**Returns**:

  XISF object.

<a id="xisf.XISF.get_images_metadata"></a>

#### get\_images\_metadata

```python
def get_images_metadata()
```

Provides the metadata of all image blocks contained in the XISF File, extracted from
the header (<Image> core elements). To get the actual image data, see read_image().

It outputs a dictionary m_i for each image, with the following structure:

```
m_i = { 
    'geometry': (dim1, ..., dimN, channels), # dim1 is the X-axis, i.e. width
    'location': (pos, size), # used internally in read_image()
    'dtype': np.dtype('...'), # derived from sampleFormat argument
    'pixelStorage': 'Planar' or 'Normal', # spec 11.5.2, defaults to 'Planar'
    'byteOrder': 'big' or 'little', # spec 10.4, defaults to 'little'
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

**Returns**:

  list [ m_0, m_1, ..., m_{n-1} ] where m_i is a dict as described above.

<a id="xisf.XISF.get_file_metadata"></a>

#### get\_file\_metadata

```python
def get_file_metadata()
```

Provides the metadata from the header of the XISF File (<Metadata> core elements).

**Returns**:

  dictionary with one entry per property: { <xisf_property_name>: property_dict, ... }
  where:
  ```
  property_dict = {'id': <xisf_property_name>, 'type': <xisf_type>, 'value': property_value, ...}
  ```

<a id="xisf.XISF.get_metadata_xml"></a>

#### get\_metadata\_xml

```python
def get_metadata_xml()
```

Returns the complete XML header as a xml.etree.ElementTree.Element object.

**Returns**:

- `xml.etree.ElementTree.Element` - complete XML XISF header

<a id="xisf.XISF.read_image"></a>

#### read\_image

```python
def read_image(n=0, data_format='channels_last')
```

Extracts an image from a XISF object.

The returned array is a read-only view on the decoded data, not a copy: it does not own its memory and cannot be modified in place. This is deliberate, since copying the array costs more than reading it (a 256 MB image measures 125 ms to read against 154 ms extra to copy). To obtain a writable array, copy it, either with ndarray.copy() or with np.array(result).

Note that the two data_format values do not have the same memory layout. Planar pixel storage decodes to (channels, \*dims) in memory, so 'channels_first' returns a C-contiguous array while 'channels_last' is a transposed view and is not C-contiguous. If you need the writable copy to be C-contiguous as well, use .copy() or np.array(result, order='C'), since np.array() alone defaults to order='K' and preserves the transposed layout.

The data block byte order is honored (spec 10.4), so blocks in either byte order are decoded, and the returned array is always in the machine's byte order. A block needing a byte swap is a copy rather than a view, and is therefore writable.

**Arguments**:

- `n` - index of the image to extract in the list returned by get_images_metadata()
- `data_format` - channels axis can be 'channels_first' or 'channels_last' (as used in
  keras/tensorflow, pyplot's imshow, etc.), 0 by default.
  

**Returns**:

  Read-only Numpy ndarray with the image data, in the requested format (channels_first or channels_last). The array does not own its memory; see the note above if you need a writable one.

<a id="xisf.XISF.read"></a>

#### read

```python
@staticmethod
def read(fname, n=0, image_metadata=None, xisf_metadata=None)
```

Convenience method for reading a file containing a single image.

**Arguments**:

- `fname` _string_ - filename
- `n` _int, optional_ - index of the image to extract (in the list returned by get_images_metadata()). Defaults to 0.
- `image_metadata` _dict, optional_ - dictionary that will be updated with the metadata of the image. If None, the image metadata is not collected, since the caller has nowhere to receive it. Defaults to None.
- `xisf_metadata` _dict, optional_ - dictionary that will be updated with the metadata of the file. If None, the file metadata is not collected, since the caller has nowhere to receive it. Defaults to None.
  

**Returns**:

- `[np.ndarray]` - Numpy ndarray with the image data, in the requested format (channels_first or channels_last). The array is a read-only view on the decoded data and cannot be modified in place; pass it to np.array() or call .copy() if you need a writable array, as read_image() does.

<a id="xisf.XISF.write"></a>

#### write

```python
@staticmethod
def write(fname, im_data, creator_app=None, image_metadata=None, xisf_metadata=None, codec=None, shuffle=False, level=None, pixel_storage=None)
```

Writes an image (numpy array) to a XISF file. Compression may be requested but it only
will be used if it actually reduces the data size.

**Arguments**:

- `fname` - filename (will overwrite if existing)
- `im_data` - numpy ndarray with the image data. Arrays of any dimensionality N >= 1 are written, with the geometry written as dim1:...:dimN:channel-count (spec 11.5.1). A 1-D or 2-D array is single-channel. A 3-D array is interpreted according to pixel_storage, so pass it explicitly if the shape is ambiguous. An array of four or more axes is written as purely spatial dimensions. The array shall be in the machine's byte order, which is the byte order written to the file (spec 10.4); an array with an explicit foreign byte order raises XISFError rather than being written with a byteOrder attribute that misdescribes it. Convert it with, for example, `im_data.astype(im_data.dtype.newbyteorder("="))`.
- `creator_app` - string for XISF:CreatorApplication file property (defaults to python version in None provided)
- `image_metadata` - dict with the same structure described for m_i in get_images_metadata().
  Only 'id', 'FITSKeywords' and 'XISFProperties' keys are actually written. The 'id' defaults to 'image' and shall match [_a-zA-Z][_a-zA-Z0-9]* (spec 11.5.2), an invalid one raises XISFError. The rest of the keys, such as 'geometry', 'sampleFormat' and 'pixelStorage', are derived from im_data and ignored.
- `xisf_metadata` - file metadata, dict with the same structure returned by get_file_metadata()
- `codec` - compression codec ('zlib', 'lz4', 'lz4hc' or 'zstd'), or None to disable compression
- `shuffle` - whether to apply byte-shuffling before compression (ignored if codec is None). Recommended
  for 'lz4' ,'lz4hc' and 'zstd' compression algorithms.
- `level` - for zlib, 1..9 (default: 6); for lz4hc, 1..12 (default: 9); for zstd, 1..22 (default: 3).
  Higher means more compression.
- `pixel_storage` - pixel storage model of im_data, using the spec naming (spec 8.5.3). 'planar' (channels first) means a (channels, \*dims) array, which is the 'channels_first' layout used by keras and TensorFlow; 'normal' (channels last) means a (\*dims, channels) array, which is the 'channels_last' layout used by numpy, matplotlib and most image tooling. When given, im_data must have at least two dimensions and the channel count is the leading axis for 'planar' or the trailing axis for 'normal'. Defaults to None, which infers the layout from the shape: a 3-D array with a trailing dimension of 1 or 3 is read as (dim1, dim2, channels),   one with a leading dimension of 1 or 3 as (channels, dim1, dim2), and anything else is treated as single-channel spatial dimensions.
- `checksum` - the cryptographic hashing algorithm used for the checksum attribute of every data block written, one of 'sha-1' (the default, and the algorithm recommended by the spec), 'sha-256', 'sha-512', 'sha3-256' or 'sha3-512'. The alternate spellings without the hyphen are accepted too. True is the same as 'sha-1', and False or None writes no checksums. The digest is computed for the serialized block, so for a compressed block it is the digest of the compressed data (spec 10.6.1), and the algorithms applied are listed in the XISF:ChecksumAlgorithms file property.

**Returns**:

- `bytes_written` - the total number of bytes written into the output file.
- `codec` - The codec actually used, i.e., None if compression did not reduce the data block size so
  compression was not finally used.

