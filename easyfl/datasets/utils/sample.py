"""
These codes are adopted from LEAF with some modifications.

Samples from all raw data;
by default samples in a non-iid manner; namely, randomly selects users from 
raw data until their cumulative amount of data exceeds the given number of 
datapoints to sample (specified by --fraction argument);
ordering of original data points is not preserved in sampled data
"""

import json
import logging
import os
import random
import time
from collections import OrderedDict

from easyfl.datasets.simulation import non_iid_class
from easyfl.datasets.utils.constants import SEED_FILES
from easyfl.datasets.utils.util import iid_divide

logger = logging.getLogger(__name__)


def extreme(data_dir, data_folder, metafile, fraction, num_class=62, num_of_client=100, class_per_client=2, seed=-1):
    """
    Note: for extreme split, there are two ways, one is divide each class into parts and then distribute to the clients;
    The second way is to let clients to go through classes to get a part of the data; Current version is the latter one, we 
    can also provide the previous one (the one we adopt in CIFA10); If (num_of_client*class_per_client)%num_class, there is no 
    difference(assume each class is equal), otherwise, how to deal with some remain parts is a question to discuss. (currently,
    the method will just give the remain part to the next client coming for collection, which may make the last clients have more
    than class_per_client;)
    """
    logger.info("------------------------------")
    logger.info("sampling data")

    subdir = os.path.join(data_dir, 'all_data')
    files = os.listdir(subdir)
    files = [f for f in files if f.endswith('.json')]

    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    logger.info("Using seed {}".format(rng_seed))
    rng = random.Random(rng_seed)

    logger.info(metafile)
    if metafile is not None:
        seed_fname = os.path.join(metafile, SEED_FILES['sampling'])
        with open(seed_fname, 'w+') as f:
            f.write("# sampling_seed used by sampling script - supply as "
                    "--smplseed to preprocess.sh or --seed to utils/sample.py\n")
            f.write(str(rng_seed))
        logger.info("- random seed written out to {file}".format(file=seed_fname))
    else:
        logger.info("- using random seed '{seed}' for sampling".format(seed=rng_seed))
    new_user_count = 0  # for iid case
    all_users = []
    all_user_data = {}
    for f in files:
        file_dir = os.path.join(subdir, f)
        with open(file_dir, 'r') as inf:
            data = json.load(inf, object_pairs_hook=OrderedDict)

        num_users = len(data['users'])

        tot_num_samples = sum(data['num_samples'])
        num_new_samples = int(fraction * tot_num_samples)

        raw_list = list(data['user_data'].values())
        raw_x = [elem['x'] for elem in raw_list]
        raw_y = [elem['y'] for elem in raw_list]
        x_list = [item for sublist in raw_x for item in sublist]  # flatten raw_x
        y_list = [item for sublist in raw_y for item in sublist]  # flatten raw_y
        num_new_users = num_users

        indices = [i for i in range(tot_num_samples)]
        new_indices = rng.sample(indices, num_new_samples)
        users = [str(i + new_user_count) for i in range(num_new_users)]
        all_users.extend(users)
        user_data = {}
        for user in users:
            user_data[user] = {'x': [], 'y': []}
        all_x_samples = [x_list[i] for i in new_indices]
        all_y_samples = [y_list[i] for i in new_indices]
        x_groups = iid_divide(all_x_samples, num_new_users)
        y_groups = iid_divide(all_y_samples, num_new_users)
        for i in range(num_new_users):
            user_data[users[i]]['x'] = x_groups[i]
            user_data[users[i]]['y'] = y_groups[i]
        all_user_data.update(user_data)

        num_samples = [len(user_data[u]['y']) for u in users]
        new_user_count += num_new_users

    allx = []
    ally = []
    for i in all_users:
        allx.extend(all_user_data[i]['x'])
        ally.extend(all_user_data[i]['y'])
    clients, all_user_data = non_iid_class(x_list, y_list, class_per_client, num_of_client)

    # ------------
    # create .json file
    all_num_samples = []
    for i in clients:
        all_num_samples.append(len(all_user_data[i]['y']))
    all_data = {}
    all_data['users'] = clients
    all_data['num_samples'] = all_num_samples
    all_data['user_data'] = all_user_data

    slabel = ''

    arg_frac = str(fraction)
    arg_frac = arg_frac[2:]
    arg_label = arg_frac
    file_name = '%s_%s_%s.json' % ("class", slabel, arg_label)
    ouf_dir = os.path.join(data_folder, 'sampled_data', file_name)

    logger.info("writing {}".format(file_name))
    with open(ouf_dir, 'w') as outfile:
        json.dump(all_data, outfile)


def sample(data_dir, data_folder, metafile, fraction, iid, iid_user_fraction=0.01, seed=-1):
    logger.info("------------------------------")
    logger.info("sampling data")
    subdir = os.path.join(data_dir, 'all_data')
    files = os.listdir(subdir)
    files = [f for f in files if f.endswith('.json')]

    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    logger.info("Using seed {}".format(rng_seed))
    rng = random.Random(rng_seed)

    logger.info(metafile)
    if metafile is not None:
        seed_fname = os.path.join(metafile, SEED_FILES['sampling'])
        with open(seed_fname, 'w+') as f:
            f.write("# sampling_seed used by sampling script - supply as "
                    "--smplseed to preprocess.sh or --seed to utils/sample.py\n")
            f.write(str(rng_seed))
        logger.info("- random seed written out to {file}".format(file=seed_fname))
    else:
        logger.info("- using random seed '{seed}' for sampling".format(seed=rng_seed))

    new_user_count = 0  # for iid case
    for f in files:
        file_dir = os.path.join(subdir, f)
        with open(file_dir, 'r') as inf:
            # Load data into an OrderedDict, to prevent ordering changes
            # and enable reproducibility
            data = json.load(inf, object_pairs_hook=OrderedDict)

        num_users = len(data['users'])

        tot_num_samples = sum(data['num_samples'])
        num_new_samples = int(fraction * tot_num_samples)

        hierarchies = None

        if iid:
            # iid in femnist is to put all data together, and then split them according to
            # iid_user_fraction * num_users numbers of clients evenly
            raw_list = list(data['user_data'].values())
            raw_x = [elem['x'] for elem in raw_list]
            raw_y = [elem['y'] for elem in raw_list]
            x_list = [item for sublist in raw_x for item in sublist]  # flatten raw_x
            y_list = [item for sublist in raw_y for item in sublist]  # flatten raw_y

            num_new_users = int(round(iid_user_fraction * num_users))
            if num_new_users == 0:
                num_new_users += 1

            indices = [i for i in range(tot_num_samples)]
            new_indices = rng.sample(indices, num_new_samples)
            users = ["f%07.0f" % (i + new_user_count) for i in range(num_new_users)]

            user_data = {}
            for user in users:
                user_data[user] = {'x': [], 'y': []}
            all_x_samples = [x_list[i] for i in new_indices]
            all_y_samples = [y_list[i] for i in new_indices]
            x_groups = iid_divide(all_x_samples, num_new_users)
            y_groups = iid_divide(all_y_samples, num_new_users)
            for i in range(num_new_users):
                user_data[users[i]]['x'] = x_groups[i]
                user_data[users[i]]['y'] = y_groups[i]

            num_samples = [len(user_data[u]['y']) for u in users]

            new_user_count += num_new_users

        else:
            # niid's fraction in femnist is to choose some clients, one by one,
            # until the data size meets the fration * total data size
            ctot_num_samples = 0

            users = data['users']
            users_and_hiers = None
            if 'hierarchies' in data:
                users_and_hiers = list(zip(users, data['hierarchies']))
                rng.shuffle(users_and_hiers)
            else:
                rng.shuffle(users)
            user_i = 0
            num_samples = []
            user_data = {}

            if 'hierarchies' in data:
                hierarchies = []

            while ctot_num_samples < num_new_samples:
                hierarchy = None
                if users_and_hiers is not None:
                    user, hier = users_and_hiers[user_i]
                else:
                    user = users[user_i]

                cdata = data['user_data'][user]

                cnum_samples = len(data['user_data'][user]['y'])

                if ctot_num_samples + cnum_samples > num_new_samples:
                    cnum_samples = num_new_samples - ctot_num_samples
                    indices = [i for i in range(cnum_samples)]
                    new_indices = rng.sample(indices, cnum_samples)
                    x = []
                    y = []
                    for i in new_indices:
                        x.append(data['user_data'][user]['x'][i])
                        y.append(data['user_data'][user]['y'][i])
                    cdata = {'x': x, 'y': y}

                if 'hierarchies' in data:
                    hierarchies.append(hier)

                num_samples.append(cnum_samples)
                user_data[user] = cdata

                ctot_num_samples += cnum_samples
                user_i += 1

            if 'hierarchies' in data:
                users = [u for u, h in users_and_hiers][:user_i]
            else:
                users = users[:user_i]

        # ------------
        # create .json file

        all_data = {}
        all_data['users'] = users
        if hierarchies is not None:
            all_data['hierarchies'] = hierarchies
        all_data['num_samples'] = num_samples
        all_data['user_data'] = user_data

        slabel = 'niid'
        if iid:
            slabel = 'iid'

        arg_frac = str(fraction)
        arg_frac = arg_frac[2:]
        arg_nu = str(iid_user_fraction)
        arg_nu = arg_nu[2:]
        arg_label = arg_frac
        if iid:
            arg_label = '%s_%s' % (arg_nu, arg_label)
        file_name = '%s_%s_%s.json' % ((f[:-5]), slabel, arg_label)
        ouf_dir = os.path.join(data_folder, 'sampled_data', file_name)

        logger.info('writing %s' % file_name)
        with open(ouf_dir, 'w') as outfile:
            json.dump(all_data, outfile)
