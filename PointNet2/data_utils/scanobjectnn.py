import os, sys, h5py, pickle, numpy as np, os.path as osp
import torch
from torch.utils.data import Dataset

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def fps(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


class ScanObjectNNHardest(Dataset):
    """The hardest variant of ScanObjectNN. 
    The data we use is: `training_objectdataset_augmentedrot_scale75.h5`[1], 
    where there are 2048 points in training and testing.
    The number of training samples is: 11416, and the number of testing samples is 2882. 
    Args:
    """
    classes = [
        "bag",
        "bin",
        "box",
        "cabinet",
        "chair",
        "desk",
        "display",
        "door",
        "shelf",
        "table",
        "bed",
        "pillow",
        "sink",
        "sofa",
        "toilet",
    ]
    num_classes = 15
    gravity_dim = 1

    def __init__(self, data_dir, split,
                 num_points=2048,
                 uniform_sample=True,
                 **kwargs):
        super().__init__()
        self.partition = split
        self.num_points = num_points
        slit_name = 'training' if split == 'train' else 'test'
        h5_name = os.path.join(
            data_dir, f'{slit_name}_objectdataset_augmentedrot_scale75.h5')

        if not osp.isfile(h5_name):
            raise FileExistsError(
                f'{h5_name} does not exist, please download dataset at first')
            
        with h5py.File(h5_name, 'r') as f:
            self.points = np.array(f['data']).astype(np.float32)
            self.labels = np.array(f['label']).astype(int)


        if slit_name == 'test' and uniform_sample:
            precomputed_path = os.path.join(
                data_dir, f'{slit_name}_objectdataset_augmentedrot_scale75_1024_fps.pkl')
            if not os.path.exists(precomputed_path):
                points = torch.from_numpy(self.points).to(torch.float32).cuda()
                self.points = fps(points, 1024).cpu().numpy()
                with open(precomputed_path, 'wb') as f:
                    pickle.dump(self.points, f)
                    print(f"{precomputed_path} saved successfully")
            else:
                with open(precomputed_path, 'rb') as f:
                    self.points = pickle.load(f)
                    print(f"{precomputed_path} load successfully")
        print(f'Successfully load ScanObjectNN {split} '
                     f'size: {self.points.shape}, num_classes: {self.labels.max()+1}')

    @property
    def num_classes(self):
        return self.labels.max() + 1

    def __getitem__(self, idx):
        current_points = self.points[idx][:self.num_points]
        label = self.labels[idx]
        if self.partition == 'train':
            np.random.shuffle(current_points)
        data = {'pos': current_points,
                'y': label
        }

        data['pos'] = pc_normalize(data['pos'])

        return data['pos'], data['y']

    def __len__(self):
        return self.points.shape[0]
