# Example file for the xisf package (https://github.com/sergio-dr/xisf)
# fits2xisf: command line tool for converting FITS to XISF
# (only the first image HDU!)

from xisf import XISF
from astropy.io import fits
import numpy as np
import argparse

APP_NAME = "fits2xisf"

help_desc = (
    "Command line tool to convert (potentially compressed) FITS files "
    "to XISF (only first image HDU!). "
    f"Based on {XISF._creator_module}."
)
parser = argparse.ArgumentParser(description=help_desc)
parser.add_argument("input_file", help="Input filename (FITS format)")
parser.add_argument("output_file", help="Output filename (XISF format)")
parser.add_argument(
    "-c",
    "--compression",
    choices=["zstd", "zlib", "lz4", "lz4hc", "none"],
    default="zstd",
    help="Compression type",
)
args = parser.parse_args()

print(f"Opening {args.input_file}...")
hdul = fits.open(args.input_file)
# Searchs for the first HDU with image data
for hdu in hdul:
    if isinstance(hdu.data, np.ndarray):
        img_data = np.atleast_3d(hdu.data)
        fits_header = hdu.header  # Header assumed in the same HDU

print(f"Image dimensions: {img_data.shape}")

print("Header:")
fits_keyw = {}
for k, v in fits_header.items():
    if not k:
        continue
    c = fits_header.comments[k]
    print(f"{k:9s}: {v} [{c}]")
    entry = fits_keyw.setdefault(k, [])
    fits_keyw[k] = entry + [{"value": str(v), "comment": str(c)}]

# Here we may attempt to populate XISF image properties from FITS headers,
# (see section '11.5.3 Astronomical Image Properties' of XISF spec.),
# but PixInsight does this anyway.
img_meta = {"FITSKeywords": fits_keyw}
codec = None if args.compression == "none" else args.compression
XISF.write(args.output_file, img_data, APP_NAME, img_meta, codec=codec, shuffle=True)
