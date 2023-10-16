# Example file for the xisf package (https://github.com/sergio-dr/xisf)
# xisf2fits: command line tool for converting XISF to FITS
# (only first image block!)

from xisf import XISF
from astropy.io import fits
import numpy as np
import argparse

APP_NAME = "xisf2fits"

help_desc = (
    "Command line tool to convert (potentially compressed) XISF files "
    "to FITS (only first image block!). "
    f"Based on {XISF._creator_module}."
)
parser = argparse.ArgumentParser(description=help_desc)
parser.add_argument("input_file", help="Input filename (XISF format)")
parser.add_argument("output_file", help="Output filename (FITS format)")
args = parser.parse_args()

print(f"Opening {args.input_file}...")
xisf = XISF(args.input_file)

img_data = np.transpose(xisf.read_image(0), (2, 0, 1))
print(f"Image dimensions: {img_data.shape}")

img_meta = xisf.get_images_metadata()[0]
print("Header:")
fits_header = []
for keyword, values in img_meta["FITSKeywords"].items():
    for value in values:
        if keyword in ("COMMENT", "HISTORY"):
            card = fits.Card(keyword, value["comment"])
        else:
            card = fits.Card(keyword, value["value"], value["comment"])
        print(card)
        fits_header.append(card)
fits_header = fits.Header(fits_header)

fits.writeto(args.output_file, img_data, fits_header, overwrite=True)
