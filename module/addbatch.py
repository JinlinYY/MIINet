from torch.utils.data import DataLoader, TensorDataset

def addbatch(data_train, data_test, batchsize):
    data = TensorDataset(data_train, data_test)
    data_loader = DataLoader(data, batch_size=batchsize, shuffle=True)
    return data_loader