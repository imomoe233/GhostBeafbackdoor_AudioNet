import os

from dataset.Dataset import Dataset


class vox2_train(Dataset):

    def __init__(self, spk_ids, root, return_file_name=False, wav_length=None):
        normalize = True
        bits = 16
        super().__init__(spk_ids, root, 'vox2_train',
                        normalize=normalize, bits=bits, 
                        return_file_name=return_file_name, wav_length=wav_length)

if __name__ == '__main__':

    from torch.utils.data import DataLoader
    root = './dataset'
    spk_ids = os.listdir('./dataset/vox2_train')
    dataset = vox2_train(spk_ids, root, return_file_name=True, wav_length=80_000)
    data_loader = DataLoader(dataset, batch_size=128, num_workers=8)

    try:
        for x, y, name in data_loader:
            print(x.shape, y, name)
    except RuntimeError as e:
        print(f"Failed to process file: {name}")
        print(e)