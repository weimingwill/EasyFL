"""
Removes users with less than the given number of samples.

These codes are adopted from LEAF with some modifications.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)


def remove(setting_folder, dataset, min_samples):
    parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    dir = os.path.join(parent_path, dataset, "data")
    subdir = os.path.join(dir, setting_folder, "sampled_data")
    files = []
    if os.path.exists(subdir):
        files = os.listdir(subdir)
    if len(files) == 0:
        subdir = os.path.join(dir, "all_data")
        files = os.listdir(subdir)
    files = [f for f in files if f.endswith(".json")]

    for f in files:
        users = []
        hierarchies = []
        num_samples = []
        user_data = {}

        file_dir = os.path.join(subdir, f)
        with open(file_dir, "r") as inf:
            data = json.load(inf)

        num_users = len(data["users"])
        for i in range(num_users):
            curr_user = data["users"][i]
            curr_hierarchy = None
            if "hierarchies" in data:
                curr_hierarchy = data["hierarchies"][i]
            curr_num_samples = data["num_samples"][i]
            if (curr_num_samples >= min_samples):
                user_data[curr_user] = data["user_data"][curr_user]
                users.append(curr_user)
                if curr_hierarchy is not None:
                    hierarchies.append(curr_hierarchy)
                num_samples.append(data["num_samples"][i])

        all_data = {}
        all_data["users"] = users
        if len(hierarchies) == len(users):
            all_data["hierarchies"] = hierarchies
        all_data["num_samples"] = num_samples
        all_data["user_data"] = user_data

        file_name = "{}_keep_{}.json".format((f[:-5]), min_samples)
        ouf_dir = os.path.join(dir, setting_folder, "rem_user_data", file_name)

        logger.info("writing {}".format(file_name))
        with open(ouf_dir, "w") as outfile:
            json.dump(all_data, outfile)
