import numpy as np

class Dataset:  # dataset class
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# it makes mini-batch with random indicies for SGD.
class BatchDataLoader:
    def __init__(self, dataset: np.ndarray, batch_size: int):
        self.batch_size = batch_size
        self.dataset = dataset
        self.indices = np.arange(len(dataset))
        np.random.shuffle(self.indices)  # for random batch

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self.dataset):
            raise StopIteration

        # make batch indices
        batch_indices = self.indices[self.current_idx:
                                     self.current_idx + self.batch_size]
        # for next iteration
        self.current_idx += self.batch_size

        '''
        if data size is not multiple of batch size, we need to handle it.
        ex) data size = 105, batch size = 10
        Then, we add extra indices to last batch.
        '''
        if len(batch_indices) < self.batch_size:
            extra_indices = batch_indices[:self.batch_size -
                                          len(batch_indices)]
            batch_indices = np.concatenate([batch_indices, extra_indices])

        # make batch data
        batch = [self.dataset[i] for i in batch_indices]
        data, labels = zip(*batch)
        return np.array(data), np.array(labels)


# run as 'python -m DL.DataLoader'
if __name__ == "__main__":
    data = np.random.rand(105, 10)
    labels = np.random.randint(0, 2, 105)
    dataset = Dataset(data, labels)
    loader = BatchDataLoader(dataset, 10)

    for x, y in loader:
        print(x.shape, y.shape)
