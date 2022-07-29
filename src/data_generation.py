from tokenize import Double
#from attr import dataclass
import numpy as np
from src.sampler import augment_sample, labels2output_map
import torch

device = 'cuda'

class ALPRDataGenerator(torch.utils.data.Dataset):
    'Generates data for PyTorch'
    def __init__(self, data, batch_size=32, dim =  128, stride = 8, shuffle=True, OutputScale = 1.0):
        'Initialization'
        super(ALPRDataGenerator, self).__init__()
        self.dim = dim
        self.stride = stride
        self.batch_size = batch_size
        self.data = data
        self.shuffle = shuffle
        self.OutputScale = OutputScale
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.data)/2)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        #indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(index)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch. Pads training data to be a multiple of batch size'
#        self.indexes = list(np.arange(0, len(self.data), 1))
        self.indexes = list(np.arange(0, len(self.data), 1)) 
        self.indexes += list(np.random.choice(self.indexes, self.batch_size - len(self.data) % self.batch_size))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, index):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Generate data

        XX, llp, ptslist = augment_sample(self.data[index][0], self.data[index][1], self.dim)
        YY = labels2output_map(llp, ptslist, self.dim, self.stride, alfa = 0.75)

        X = XX*self.OutputScale

        X = torch.from_numpy(XX).permute(2, 0, 1).to(device)
        y = torch.from_numpy(YY).to(device)

        return X, y



