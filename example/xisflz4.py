# Example file for the xisf package (https://github.com/sergio-dr/xisf)
# xisflz4: command line tool for compressing XISF files (only first image block!)

from xisf import XISF
import argparse

APP_NAME = "xisflz4"

help_desc = (
    "Command line tool for compressing XISF files (only first image block!) "
    f"with LZ4HC codec and byte shuffling. Based on {XISF._creator_module}."
)
parser = argparse.ArgumentParser(description=help_desc)
parser.add_argument("input_file", help="Input filename (XISF format)")
parser.add_argument("output_file", help="Output filename (XISF format)")
args = parser.parse_args()

xisf = XISF(args.input_file)

img_data = xisf.read_image(0)
img_meta = xisf.get_images_metadata()[0]
file_meta = xisf.get_file_metadata()

bw, codec = XISF.write(
    args.output_file, img_data, APP_NAME, img_meta, file_meta, "lz4hc", True
)
print(f"{bw} bytes written.")
if not codec:
    print("No compression used as the uncompressed data block is smaller.")
