# Example file for the xisf package (https://github.com/sergio-dr/xisf)
# xisfmeta: command line tool for printing XISF metadata

from xisf import XISF
import argparse

help_desc = "Command line tool for printing XISF metadata"
parser = argparse.ArgumentParser(description=help_desc)
parser.add_argument("input_file", help="Input filename (XISF format)")
args = parser.parse_args()

xisf = XISF(args.input_file)
file_meta = xisf.get_file_metadata()    
ims_meta = xisf.get_images_metadata()

print(f"Filename: {args.input_file}")

print("\n\n__/ File metadata \__________")
for key, value in file_meta.items():
    print(f"{key:30s}: {value}")

pseudokey = "# of images"
print(f"{pseudokey:30s}: {len(ims_meta)}")


def crop(text, max=32):
    if len(text) > max:
        return f"{text[:max]} [...]"
    else:
        return text


for i, im_meta in enumerate(ims_meta):
    print(f"\n\n__/ Image #{i} \__________")

    # Simple values
    for key, val in im_meta.items():
        if not isinstance(val, dict):
            print(f"{key:30s}: {val}")

    # XISFProperties (dict)
    key = 'XISFProperties'
    print(f"{key:30s}: ")
    xisf_meta = im_meta[key]
    for xisf_keyw, xisf_val in xisf_meta.items():
        print(f"\t{xisf_keyw:36s}: {crop(xisf_val)}")

    # FITSKeywords: { '<keyword>': [ {'value': ..., 'comment': ...}, ...], 
    #                 ... },
    key = 'FITSKeywords'
    print(f"{key:30s}: ")
    fits_meta = im_meta[key]
    for j, (fits_keyw, fits_val) in enumerate(fits_meta.items()):
        if len(fits_val) == 1:
            print(f"\t{fits_keyw:14s}: {fits_val[0]['value']} [{fits_val[0].get('comment', '-')}]")
        else:
            for k, fits_val_k in enumerate(fits_val):
                key = f"{fits_keyw}[{k:4d}]"
                print(f"\t{key:14s}: {fits_val_k['value']} [{fits_val_k.get('comment', '-')}]")
