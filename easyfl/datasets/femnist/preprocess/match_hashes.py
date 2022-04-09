"""
These codes are adopted from LEAF with some modifications.
"""
import os

from easyfl.datasets.utils import util


def match_hash(base_folder):
    cfhd = os.path.join(base_folder, "intermediate", "class_file_hashes")
    wfhd = os.path.join(base_folder, "intermediate", "write_file_hashes")
    class_file_hashes = util.load_obj(cfhd)
    write_file_hashes = util.load_obj(wfhd)
    class_hash_dict = {}
    for i in range(len(class_file_hashes)):
        (c, f, h) = class_file_hashes[len(class_file_hashes) - i - 1]
        class_hash_dict[h] = (c, f)

    write_classes = []
    for tup in write_file_hashes:
        (w, f, h) = tup
        write_classes.append((w, f, class_hash_dict[h][0]))

    wwcd = os.path.join(base_folder, "intermediate", "write_with_class")
    util.save_obj(write_classes, wwcd)
