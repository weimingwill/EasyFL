"""
These codes are adopted from LEAF with some modifications.
"""

import json
import os

from easyfl.datasets.shakespeare.utils.shake_utils import parse_data_in


def generated_all_data(parent_path):
    users_and_plays_path = os.path.join(parent_path, 'raw_data', 'users_and_plays.json')
    txt_dir = os.path.join(parent_path, 'raw_data', 'by_play_and_character')
    json_data = parse_data_in(txt_dir, users_and_plays_path)
    json_path = os.path.join(parent_path, 'all_data', 'all_data.json')
    with open(json_path, 'w') as outfile:
        json.dump(json_data, outfile)
