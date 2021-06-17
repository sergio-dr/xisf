# XISF for python

mplements an incomplete Baseline XISF Decoder. It parses file and attached images metadata. Image data is returned as a numpy ndarray, using the "channels-last" convention. 

What's supported: 
- Monolithic XISF files only
  - XISF blocks with attachment block locations
  - Planar pixel storage models, *however it assumes 2D images only* (with multiple channels)
  - UInt8/16/32 and Float32/64 pixel sample formats
  - Grayscale and RGB color spaces     
- Decoding:
  - multiple Image core elements from a monolithic XISF file
  - Support all standard compression codecs defined in this specification for decompression (zlib/lz4[hc]+ byte shuffling)
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
```python
>>> from xisf import XISF
>>> import matplotlib.pyplot as plt
>>> xisf = XISF("file.xisf")
>>> file_meta = xisf.get_file_metadata()    
>>> file_meta
>>> ims_meta = xisf.get_images_metadata()
>>> ims_meta
>>> im_data = xisf.read_image(0)
>>> plt.imshow(im_data)
>>> XISF.write("output.xisf", im_data, ims_meta[0], file_meta)
```

The XISF format specification is available at https://pixinsight.com/doc/docs/XISF-1.0-spec/XISF-1.0-spec.html
