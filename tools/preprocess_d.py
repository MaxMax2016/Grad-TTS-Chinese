import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
from torch.utils.data import DataLoader
from grad_extend.data import TextMelDataset, TextMelBatchCollate


filelist_path = "./data/files/valid.txt"

dataset = TextMelDataset(filelist_path)
collate = TextMelBatchCollate()
loader = DataLoader(dataset=dataset, 
                    batch_size=2,
                    collate_fn=collate, 
                    drop_last=True,
                    num_workers=1, 
                    shuffle=True)

for batch in tqdm(loader):
    # {'x': x, 'x_lengths': x_lengths, 'b': b, 'y': y, 'y_lengths': y_lengths}
    print('x', batch['x'].shape)
    print('x_lengths', batch['x_lengths'].shape)
    print('b', batch['b'].shape)
    print('y', batch['y'].shape)
    print('y_lengths', batch['y_lengths'].shape)
