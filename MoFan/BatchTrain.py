import torch
import torch.utils.data as Data


torch.manual_seed(100)

BATCH_SIZE = 5

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training
    num_workers=0,              # subprocesses for loading data
)

# train entire dataset 3 times
for epoch in range(3):
    for i, (batch_x, batch_y) in enumerate(loader):  # for each training step
        # train process...
        print('Epoch: ', epoch, '| Step: ', i, '| batch x: ',
              batch_x.numpy(), '| batch y: ', batch_y.numpy())
