"""
These codes are adopted from LEAF with some modifications.
"""

import hashlib
import logging
import os

from easyfl.datasets.utils import util

logger = logging.getLogger(__name__)


def get_hash(base_folder):
    cfd = os.path.join(base_folder, "intermediate", "class_file_dirs")
    wfd = os.path.join(base_folder, "intermediate", "write_file_dirs")
    class_file_dirs = util.load_obj(cfd)
    write_file_dirs = util.load_obj(wfd)

    class_file_hashes = []
    write_file_hashes = []

    count = 0
    for tup in class_file_dirs:
        if (count % 100000 == 0):
            logger.info("hashed %d class images" % count)

        (cclass, cfile) = tup
        file_path = os.path.join(base_folder, cfile)

        chash = hashlib.md5(open(file_path, "rb").read()).hexdigest()

        class_file_hashes.append((cclass, cfile, chash))

        count += 1

    cfhd = os.path.join(base_folder, "intermediate", "class_file_hashes")
    util.save_obj(class_file_hashes, cfhd)

    count = 0
    for tup in write_file_dirs:
        if (count % 100000 == 0):
            logger.info("hashed %d write images" % count)

        (cclass, cfile) = tup
        file_path = os.path.join(base_folder, cfile)

        chash = hashlib.md5(open(file_path, "rb").read()).hexdigest()

        write_file_hashes.append((cclass, cfile, chash))

        count += 1

    wfhd = os.path.join(base_folder, "intermediate", "write_file_hashes")
    util.save_obj(write_file_hashes, wfhd)
