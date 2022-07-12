import os, cv2
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
# from utils import data_utils
from oss import OssProxy, OssFile
from io import BytesIO




def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class InferenceDataset(Dataset):

    def __init__(self, root, opts, transform=None, preprocess=None):
        self.paths = sorted(data_utils.make_dataset(root))
        self.transform = transform
        self.preprocess = preprocess
        self.opts = opts

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from_path = self.paths[index]
        if self.preprocess is not None:
            from_im = self.preprocess(from_path)
        else:
            from_im = Image.open(from_path).convert('RGB')
        if self.transform:
            from_im = self.transform(from_im)
        return from_im


class InferenceDataset_causal(Dataset):

    def __init__(self, anno_file, opts, transform=None, preprocess=None, delimiter='||$$||'):
        self.opts = opts
        self.transform = transform
        self.preprocess = preprocess
        self.oss_proxy = OssProxy()
        if os.path.isfile(anno_file):
            with open(anno_file, 'r') as f:
                self.anno = f.readlines()
        else:
            if 'http' in anno_file:
                with urllib.request.urlopen(anno_file) as f:
                    self.anno = [str(line.decode('utf-8')).strip('\n') for line in f]
            else:
                with OssFile(anno_file).get_str_file() as f:
                    self.anno = f.readlines()

        self.images = [tt.strip().split(delimiter)[-2] for tt in self.anno]
        self.targets = [int(tt.strip().split(delimiter)[-1]) for tt in self.anno]
        self.images = self.images[:128*32]
        self.targets = self.targets[:128*32]

        print('self.images: {}'.format(len(self.images)))
        print('self.targets: {}'.format(len(self.targets)))
        for i in range(10):
            print(self.images[i], self.targets[i])
        print('self.targets - pos: {}'.format(len([i for i in self.targets if i==1])))
        print('self.targets - neg: {}'.format(len([i for i in self.targets if i==0])))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.images[index]
        target = self.targets[index]

        if os.path.isfile(path):
            img = pil_loader(path)
        else:
            img = self.oss_loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return path, img, target

    def oss_loader(self, img_path):
        img = None
        for _ in range(10):  # try 10 times
            try:
                data = self.oss_proxy.download_to_bytes(img_path)
                temp_buffer = BytesIO()
                temp_buffer.write(data)
                temp_buffer.seek(0)
                img = np.fromstring(temp_buffer.getvalue(), np.uint8)
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                temp_buffer.close()
            except Exception as err:
                print('load image error:', img_path, err)
            if img is not None:
                break
        return img


if __name__=='__main__':
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        )
    test_dataset = InferenceDataset_causal(anno_file='leogb/causal_logs/CelebAMask-HQ-9_v2/train_attrbute_9.txt',
                                    transform=transform,
                                    preprocess=None,
                                    opts={})
    data_loader = DataLoader(test_dataset,
                             batch_size=32,
                             shuffle=False,
                             num_workers=32,
                             drop_last=True)
    for i, (batch_path, batch_img, batch_target) in enumerate(data_loader):
   	    print(i, len(batch_path), len(batch_img), len(batch_target))
   	    break

