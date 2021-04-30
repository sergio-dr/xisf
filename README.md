# xisf

Implements an *uncomplete* XISF (Extensible Image Serialization Format) Decoder. It parses file and attached images metadata. Image data is returned as a numpy ndarray, using the "channels last" convention. 

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
```python
>>> from xisf import XISF
>>> import matplotlib.pyplot as plt
>>> xisf = XISF()
>>> xisf.read("file.xisf")
>>> ims_meta = xisf.get_images_metadata()
>>> ims_meta
>>> im_data = xisf.read_image(0)
>>> plt.imshow(im_data)
>>> xisf.close()
```

The XISF format specification is available at https://pixinsight.com/doc/docs/XISF-1.0-spec/XISF-1.0-spec.html
