# Example file for the xisf package (https://github.com/sergio-dr/xisf)
# xisfstripes: writes a synthetic RGB XISF image that is easy to check by eye
# in PixInsight, to verify that geometry, axis order and channel order are
# written correctly.

from xisf import XISF
import numpy as np
import argparse

APP_NAME = "xisfstripes"

help_desc = (
    "Command line tool to write a synthetic RGB XISF image composed of "
    "alternating red, green and blue stripes. Useful to check by eye that "
    f"images are written with the correct orientation. Based on {XISF._creator_module}."
)
parser = argparse.ArgumentParser(description=help_desc)
parser.add_argument(
    "output_file", nargs="?", default="stripes.xisf", help="Output filename (XISF format)"
)
parser.add_argument("-w", "--width", type=int, default=600, help="Image width in pixels")
parser.add_argument("-H", "--height", type=int, default=400, help="Image height in pixels")
parser.add_argument(
    "-s", "--stripe-height", type=int, default=10, help="Height of each stripe in pixels"
)
parser.add_argument(
    "--dtype",
    default="uint16",
    choices=["uint8", "uint16", "uint32", "float32"],
    help="Sample format of the written image",
)
args = parser.parse_args()

# The image is written as a numpy array of shape (height, width, channels),
# i.e. channels last. The stripes are horizontal bands stacked from top to
# bottom, so the topmost stripe is red and the bottommost is blue. Since a
# 10-pixel-high stripe spanning the full width is a band, the colors cycle
# with increasing Y, which makes a flipped or transposed image obvious.
n_stripes, remainder = divmod(args.height, args.stripe_height)
if n_stripes < 1:
    parser.error(
        f"stripe height {args.stripe_height} must not exceed height {args.height}"
    )
if remainder:
    print(
        f"Warning: {args.height} is not a multiple of {args.stripe_height}, the"
        f" remaining {remainder} rows will be left black."
    )

im_data = np.zeros((args.height, args.width, 3), dtype=getattr(np, args.dtype))
# Lit channels are written at full scale, so that the stripes are the brightest
# possible color in the sample format. A float image gets the bounds 0:1 that
# the writer assumes, so 1.0 is full scale there and the dtype maximum otherwise.
full_scale = 1.0 if np.issubdtype(im_data.dtype, np.floating) else np.iinfo(im_data.dtype).max
COLORS = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
for i in range(n_stripes):
    channel = np.array(COLORS[i % 3], dtype=im_data.dtype) * full_scale
    im_data[i * args.stripe_height : (i + 1) * args.stripe_height, :, :] = channel

img_meta = {"FITSKeywords": {"COLORSPC": [{"value": "RGB", "comment": "Stripes"}]}}
XISF.write(
    args.output_file,
    im_data,
    APP_NAME,
    img_meta,
    pixel_storage="normal",
)

print(f"Wrote {args.output_file}: {args.width}x{args.height}, 3 channels, RGB.")
print(f"Geometry is {args.width}:{args.height}:3 (width:height:channels).")
print(f"{n_stripes} stripes of {args.stripe_height}px, red, green, blue, from the top down.")

# Read the file back and check that what comes out matches what went in, so that
# any orientation mistake is reported here rather than only being visible in
# PixInsight.
xisf = XISF(args.output_file)
metadata = xisf.get_images_metadata()[0]
back = xisf.read_image(0)
print(f"Read back geometry {':'.join(map(str, metadata['geometry']))}, shape {back.shape}.")
if back.shape != im_data.shape or not np.array_equal(back, im_data):
    raise SystemExit("ERROR: the image read back does not match the image written.")
print("Round-trip OK: the image read back matches the image written.")
