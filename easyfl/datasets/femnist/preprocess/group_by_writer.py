"""
These codes are adopted from LEAF with some modifications.
"""
import os

from easyfl.datasets.utils import util


def group_by_writer(base_folder):
    wwcd = os.path.join(base_folder, "intermediate", "write_with_class")
    write_class = util.load_obj(wwcd)

    writers = []  # each entry is a (writer, [list of (file, class)]) tuple
    cimages = []
    (cw, _, _) = write_class[0]
    for (w, f, c) in write_class:
        if w != cw:
            writers.append((cw, cimages))
            cw = w
            cimages = [(f, c)]
        cimages.append((f, c))
    writers.append((cw, cimages))

    ibwd = os.path.join(base_folder, "intermediate", "images_by_writer")
    util.save_obj(writers, ibwd)
