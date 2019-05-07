import numpy as np
from scipy.misc import imsave
from os import listdir, makedirs
from os.path import isfile, join
import sys, getopt

from SALICONtf import SALICONtf


def main(argv):
    img_dir = ''
    out_dir = ''

    if argv:
        opts, args = getopt.getopt(argv, "i:o:w:")
    else:
        print('Usage: python3 run_SALICON.py -w <model_weights> -i <input_dir> -o <output_dir>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-i':
            img_dir = arg
        elif opt == '-o':
            out_dir = arg
        elif opt == '-w':
            model_weights = arg

    makedirs(out_dir, exist_ok=True)

    images = [f for f in listdir(img_dir) if isfile(join(img_dir, f))]

    s = SALICONtf(model_weights)

    for img_name in images:
        smap = s.compute_saliency(img_path=join(img_dir, img_name))
        imsave(join(out_dir, img_name), smap)


if __name__ == "__main__":
    main(sys.argv[1:])
