import os
import os.path
import random
import warnings

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from easyfl.datasets import FederatedTorchDataset

DEFAULT_TASKS = ['depth_zbuffer', 'normal', 'segment_semantic', 'edge_occlusion', 'reshading', 'keypoints2d', 'edge_texture']


VAL_LIMIT = 100
TEST_LIMIT = (1000, 2000)


def get_dataset(data_dir, train_client_file, test_client_file, tasks, image_size, model_limit=None, half_sized_output=False, augment=False):
    dataset = {}  # each building in taskonomy dataset is a client
    client_ids = set()
    with open(train_client_file) as f:
        for line in f:
            client_id = line.strip()
            client_ids.add(client_id)
            dataset[client_id] = TaskonomyLoader(data_dir,
                                                 label_set=tasks,
                                                 model_whitelist=[client_id],
                                                 model_limit=model_limit,
                                                 output_size=(image_size, image_size),
                                                 half_sized_output=half_sized_output,
                                                 augment=augment)
            print(f'Client {client_id}: {len(dataset[client_id])} instances.')
    train_set = FederatedTorchDataset(dataset, client_ids)

    if augment == "aggressive":
        print('Data augmentation is on (aggressive).')
    elif augment:
        print('Data augmentation is on (flip).')
    else:
        print('no data augmentation')

    test_client_ids = set()
    with open(test_client_file) as f:
        for line in f:
            test_client_ids.add(line.strip())

    val_set = get_validation_data(data_dir, test_client_ids, tasks, image_size, VAL_LIMIT, half_sized_output)
    test_set = get_validation_data(data_dir, test_client_ids, tasks, image_size, TEST_LIMIT, half_sized_output)

    return train_set, val_set, test_set


def get_validation_data(data_dir, client_ids, tasks, image_size, model_limit, half_sized_output=False):
    dataset = TaskonomyLoader(data_dir,
                              label_set=tasks,
                              model_whitelist=client_ids,
                              model_limit=model_limit,
                              output_size=(image_size, image_size),
                              half_sized_output=half_sized_output,
                              augment=False)
    if model_limit == VAL_LIMIT:
        print(f'Found {len(dataset)} validation instances.')
    else:
        print(f'Found {len(dataset)} test instances.')
    return FederatedTorchDataset(dataset, client_ids)


class TaskonomyLoader(data.Dataset):
    def __init__(self,
                 root,
                 label_set=DEFAULT_TASKS,
                 model_whitelist=None,
                 model_limit=None,
                 output_size=None,
                 convert_to_tensor=True,
                 return_filename=False,
                 half_sized_output=False,
                 augment=False):
        self.root = root
        self.model_limit = model_limit
        self.records = []
        if model_whitelist is None:
            self.model_whitelist = None
        elif type(model_whitelist) is str:
            self.model_whitelist = set()
            with open(model_whitelist) as f:
                for line in f:
                    self.model_whitelist.add(line.strip())
        else:
            self.model_whitelist = model_whitelist

        for i, (where, subdirs, files) in enumerate(os.walk(os.path.join(root, 'rgb'))):
            if subdirs:
                continue
            model = where.split('/')[-1]
            if self.model_whitelist is None or model in self.model_whitelist:
                full_paths = [os.path.join(where, f) for f in files]
                if isinstance(model_limit, tuple):
                    full_paths.sort()
                    full_paths = full_paths[model_limit[0]:model_limit[1]]
                elif model_limit is not None:
                    full_paths.sort()
                    full_paths = full_paths[:model_limit]
                self.records += full_paths

        # self.records = manager.list(self.records)
        self.label_set = label_set
        self.output_size = output_size
        self.half_sized_output = half_sized_output
        self.convert_to_tensor = convert_to_tensor
        self.return_filename = return_filename
        self.to_tensor = transforms.ToTensor()
        self.augment = augment

        self.last = {}

    def process_image(self, im, input=False):
        output_size = self.output_size
        if self.half_sized_output and not input:
            if output_size is None:
                output_size = (128, 128)
            else:
                output_size = output_size[0] // 2, output_size[1] // 2
        if output_size is not None and output_size != im.size:
            im = im.resize(output_size, Image.BILINEAR)

        bands = im.getbands()
        if self.convert_to_tensor:
            if bands[0] == 'L':
                im = np.array(im)
                im.setflags(write=1)
                im = torch.from_numpy(im).unsqueeze(0)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    im = self.to_tensor(im)

        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is an uint8 matrix of integers with the same width and height.
        If there is an error loading an image or its labels, simply return the previous example.
        """
        with torch.no_grad():
            file_name = self.records[index]
            save_filename = file_name

            flip_lr = (random.randint(0, 1) > .5 and self.augment)

            flip_ud = (random.randint(0, 1) > .5 and (self.augment == "aggressive"))

            pil_im = Image.open(file_name)

            if flip_lr:
                pil_im = ImageOps.mirror(pil_im)
            if flip_ud:
                pil_im = ImageOps.flip(pil_im)

            im = self.process_image(pil_im, input=True)

            error = False

            ys = {}
            mask = None
            to_load = self.label_set
            if len(set(['edge_occlusion', 'normal', 'reshading', 'principal_curvature']).intersection(
                    self.label_set)) != 0:
                if os.path.isfile(file_name.replace('rgb', 'mask')):
                    to_load.append('mask')
                elif 'depth_zbuffer' not in to_load:
                    to_load.append('depth_zbuffer')

            for i in to_load:
                if i == 'mask' and mask is not None:
                    continue

                yfilename = file_name.replace('rgb', i)
                try:
                    yim = Image.open(yfilename)
                except:
                    yim = self.last[i].copy()
                    error = True
                if (i in self.last and yim.getbands() != self.last[i].getbands()) or error:
                    yim = self.last[i].copy()
                try:
                    self.last[i] = yim.copy()
                except:
                    pass
                if flip_lr:
                    try:
                        yim = ImageOps.mirror(yim)
                    except:
                        pass
                if flip_ud:
                    try:
                        yim = ImageOps.flip(yim)
                    except:
                        pass
                try:
                    yim = self.process_image(yim)
                except:
                    yim = self.last[i].copy()
                    yim = self.process_image(yim)

                if i == 'depth_zbuffer':
                    yim = yim.float()
                    mask = yim < (2 ** 13)
                    yim -= 1500.0
                    yim /= 1000.0
                elif i == 'edge_occlusion':
                    yim = yim.float()
                    yim -= 56.0248
                    yim /= 239.1265
                elif i == 'keypoints2d':
                    yim = yim.float()
                    yim -= 50.0
                    yim /= 100.0
                elif i == 'edge_texture':
                    yim = yim.float()
                    yim -= 718.0
                    yim /= 1070.0
                elif i == 'normal':
                    yim = yim.float()
                    yim -= .5
                    yim *= 2.0
                    if flip_lr:
                        yim[0] *= -1.0
                    if flip_ud:
                        yim[1] *= -1.0
                elif i == 'reshading':
                    yim = yim.mean(dim=0, keepdim=True)
                    yim -= .4962
                    yim /= 0.2846
                    # print('reshading',yim.shape,yim.max(),yim.min())
                elif i == 'principal_curvature':
                    yim = yim[:2]
                    yim -= torch.tensor([0.5175, 0.4987]).view(2, 1, 1)
                    yim /= torch.tensor([0.1373, 0.0359]).view(2, 1, 1)
                    # print('principal_curvature',yim.shape,yim.max(),yim.min())
                elif i == 'mask':
                    mask = yim.bool()
                    yim = mask

                ys[i] = yim

            if mask is not None:
                ys['mask'] = mask

            if not 'rgb' in self.label_set:
                ys['rgb'] = im

            if self.return_filename:
                return im, ys, file_name
            else:
                return im, ys

    def __len__(self):
        return len(self.records)


class DataPrefetcher:
    def __init__(self, loader, device):
        self.inital_loader = loader
        self.device = device
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            # self.next_input = None
            # self.next_target = None
            self.loader = iter(self.inital_loader)
            self.preload()
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.to(self.device, non_blocking=True)
            # self.next_target = self.next_target.cuda(async=True)
            self.next_target = {key: val.to(self.device, non_blocking=True) for (key, val) in self.next_target.items()}

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target

    def update_device(self, device):
        self.device = device


def show(im, ys):
    from matplotlib import pyplot as plt
    plt.figure(figsize=(30, 30))
    plt.subplot(4, 3, 1).set_title('RGB')
    im = im.permute([1, 2, 0])
    plt.imshow(im)
    for i, y in enumerate(ys):
        yim = ys[y]
        plt.subplot(4, 3, 2 + i).set_title(y)
        if y == 'normal':
            yim += 1
            yim /= 2
        if yim.shape[0] == 2:
            yim = torch.cat([yim, torch.zeros((1, yim.shape[1], yim.shape[2]))], dim=0)
        yim = yim.permute([1, 2, 0])
        yim = yim.squeeze()
        plt.imshow(np.array(yim))

    plt.show()


def test():
    loader = TaskonomyLoader(
        '/home/tstand/Desktop/lite_taskonomy/',
        label_set=['normal', 'reshading', 'principal_curvature', 'edge_occlusion', 'depth_zbuffer'],
        augment='aggressive')

    totals = {}
    totals2 = {}
    count = {}
    indices = list(range(len(loader)))
    random.shuffle(indices)
    for data_count, index in enumerate(indices):
        im, ys = loader[index]
        show(im, ys)
        mask = ys['mask']
        # mask = ~mask
        print(index)
        for i, y in enumerate(ys):
            yim = ys[y]
            yim = yim.float()
            if y not in totals:
                totals[y] = 0
                totals2[y] = 0
                count[y] = 0

            totals[y] += (yim * mask).sum(dim=[1, 2])
            totals2[y] += ((yim ** 2) * mask).sum(dim=[1, 2])
            count[y] += (torch.ones_like(yim) * mask).sum(dim=[1, 2])

            # print(y,yim.shape)
            std = torch.sqrt((totals2[y] - (totals[y] ** 2) / count[y]) / count[y])
            print(data_count, '/', len(loader), y, 'mean:', totals[y] / count[y], 'std:', std)


def output_mask(index, loader):
    filename = loader.records[index]
    filename = filename.replace('rgb', 'mask')
    filename = filename.replace('/intel_nvme/taskonomy_data/', '/run/shm/')
    if os.path.isfile(filename):
        return

    print(filename)

    x, ys = loader[index]

    mask = ys['mask']
    mask = mask.squeeze()
    mask_im = Image.fromarray(mask.numpy())
    mask_im = mask_im.convert(mode='1')
    # plt.subplot(2,1,1)
    # plt.imshow(mask)
    # plt.subplot(2,1,2)
    # plt.imshow(mask_im)
    # plt.show()
    path, _ = os.path.split(filename)
    os.makedirs(path, exist_ok=True)
    mask_im.save(filename, bits=1, optimize=True)


def get_masks():
    loader = TaskonomyLoader(
        '/intel_nvme/taskonomy_data/',
        label_set=['depth_zbuffer'],
        augment=False)

    indices = list(range(len(loader)))

    random.shuffle(indices)

    for count, index in enumerate(indices):
        print(count, len(indices))
        output_mask(index, loader)


if __name__ == "__main__":
    file_name = "/Users/weiming/personal-projects/taskonomy_dataset/rgb/cosmos/point_512_view_7_domain_rgb.png"

    pil_im = Image.open(file_name)

    pil_im = ImageOps.mirror(pil_im)

    output_size = (128, 128)
    pil_im = pil_im.resize(output_size, Image.BILINEAR)

    print(pil_im)
    print("Completed")
