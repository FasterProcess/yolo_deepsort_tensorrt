import torch
from torchvision.transforms import transforms
import numpy as np
import cv2
import os

class CalibratorDataset:
    def __init__(
        self,
        calibration_image_folder,
        input_shape=(-1, 3, 224, 224),
        batch_size=1,
        skip_frame=1,
        dataset_limit=1 * 1000,
    ):
        self.image_folder = calibration_image_folder

        self.preprocess_flag = True
        self.datasets = None
        self.datasize = 0

        self.dataset_limit = dataset_limit
        self.skip_frame = skip_frame

        (_, _, self.height, self.width) = input_shape
        self.batch_size = batch_size
        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.init_data()

    def __len__(self) -> int:
        return self.datasize

    def shape(self) -> tuple:
        return self.datasets[0].shape

    def __getitem__(self, index) -> np.ndarray:
        if index < self.datasize:
            return self.datasets[index]
        else:
            return None

    def init_data(self):
        self.preprocess_flag = False
        self.datasets = self.load_pre_data(
            self.image_folder, size_limit=self.dataset_limit, skip=self.skip_frame
        )  # (k*b+m,c,h,w)
        self.datasize = (
            len(self.datasets) // self.batch_size * self.batch_size // self.batch_size
        )

        self.datasets = np.split(
            self.datasets[: self.datasize * self.batch_size, ...],
            self.datasize,
            axis=0,
        )

        self.datasets = [np.ascontiguousarray(data) for data in self.datasets]
        print(
            f"finish init calibration in cpu: datasize={len(self)}*{self.shape()}, type={self.datasets[0].dtype}"
        )

    def preprocess(self, np_image: np.ndarray, bgr=True):
        """
        Preprocessing for embedder network: Flips BGR to RGB, resize, convert to torch tensor, normalise with imagenet mean and variance, reshape. Note: input image yet to be loaded to GPU through tensor.cuda()

        Parameters
        ----------
        np_image : ndarray
            (H x W x C)

        Returns
        -------
        Torch Tensor

        """
        if bgr:
            np_image_rgb = np_image[..., ::-1]
        else:
            np_image_rgb = np_image

        input_image = cv2.resize(np_image_rgb, (self.width, self.height))
        # trans = transforms.Compose(
        #     [
        #         transforms.ToTensor(),
        #         transforms.Normalize(
        #             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        #         ),
        #     ]
        # )
        input_image = self.norm(
            (torch.from_numpy(input_image) / 255.0).moveaxis(-1, 0)
        )  # type: torch.Tensor
        # input_image = input_image.view(
        #     1, 3, self.height, self.width
        # )  # type:torch.Tensor

        return input_image.cpu().numpy()

    def load_pre_data(self, imgs_folder, size_limit=0, skip=20):
        img_names = os.listdir(imgs_folder)
        imgs = []
        idx = -1
        for img_name in img_names:
            img_path = os.path.join(imgs_folder, img_name)
            img = cv2.imread(img_path)
            idx += 1

            if idx % skip == 0:
                print(f"load {img_path}")
                imgs.append(self.preprocess(img))
                idx = 0

            if size_limit > 0 and len(imgs) >= size_limit:
                break

        assert len(imgs) > 0, "empty datas"

        return np.stack(imgs)