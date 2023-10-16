# Example file for the xisf package (https://github.com/sergio-dr/xisf)
# xisfmeta: command line tool for printing XISF metadata

from xisf import XISF
import argparse
import xml.etree.ElementTree as ET

TAB = " " * 4


# In python 3.9+, use ET.indent(...) instead of these:
#   https://stackoverflow.com/a/65808327
def _pretty_print(current, parent=None, index=-1, depth=0):
    for i, node in enumerate(current):
        _pretty_print(node, current, i, depth + 1)
    if parent is not None:
        if index == 0:
            parent.text = "\n" + (TAB * depth)
        else:
            parent[index - 1].tail = "\n" + (TAB * depth)
        if index == len(parent) - 1:
            current.tail = "\n" + (TAB * (depth - 1))


def prettify(etree, depth=0):
    _pretty_print(etree, depth=depth)
    return TAB * depth + ET.tostring(etree, encoding="unicode", xml_declaration=True)


help_desc = (
    f"Command line tool for printing XISF metadata. Based on {XISF._creator_module}."
)
parser = argparse.ArgumentParser(description=help_desc)
parser.add_argument("input_file", help="Input filename (XISF format)")
parser.add_argument(
    "-x", "--xml", action="store_true", help="Outputs pretty-printed XML header"
)
args = parser.parse_args()

xisf = XISF(args.input_file)

if args.xml:
    print(prettify(xisf.get_metadata_xml()))
else:
    file_meta = xisf.get_file_metadata()
    ims_meta = xisf.get_images_metadata()

    print(f"Filename: {args.input_file}")

    print("\n\n__/ File metadata \__________")
    for key, prop in file_meta.items():
        print(f"{key:30s} [{prop['type']:10s}]: {prop['value']}")

    pseudokey = "(# of images)"
    print(f"{pseudokey:43s}: {len(ims_meta)}")

    def render(value, width=40):
        text = str(value)
        if len(text) > width:
            # Print indented text starting the next line
            if text.startswith("<?xml"):
                return "\n" + prettify(ET.fromstring(text), depth=2)
            else:
                indented = "\n"
                wide = 80
                for line in text.splitlines(True):
                    for i in range(0, len(line), wide):
                        indented += TAB * 2 + line[i : i + wide] + "\n"
                return indented
        else:
            # Short text, print on the same line
            return text

    for i, im_meta in enumerate(ims_meta):
        print(f"\n\n__/ Image #{i} \__________")

        # Image attributes
        key = "Image attributes"
        print(f"{key:30s}: ")
        for key, val in im_meta.items():
            if not isinstance(val, dict):
                print(f"{TAB}{key:30s}: {val}")

        # XISFProperties (dict)
        key = "XISFProperties"
        print(f"{key:30s}: ")
        xisf_meta = im_meta[key]
        for xisf_keyw, xisf_prop in xisf_meta.items():
            props_str = ", ".join(
                [f"{k}: {v}" for k, v in xisf_prop.items() if k not in ("id", "value")]
            )
            print(f"{TAB}{xisf_keyw:36s} [{props_str}]: {render(xisf_prop['value'])}")

        # FITSKeywords: { '<keyword>': [ {'value': ..., 'comment': ...}, ...],
        #                 ... },
        key = "FITSKeywords"
        print(f"{key:30s}: ")
        fits_meta = im_meta[key]
        for j, (fits_keyw, fits_val) in enumerate(fits_meta.items()):
            if len(fits_val) == 1:
                print(
                    f"{TAB}{fits_keyw:14s}: {fits_val[0]['value']} [{fits_val[0].get('comment', '-')}]"
                )
            else:
                for k, fits_val_k in enumerate(fits_val):
                    key = f"{fits_keyw}[{k:4d}]"
                    print(
                        f"{TAB}{key:14s}: {fits_val_k['value']} [{fits_val_k.get('comment', '-')}]"
                    )
