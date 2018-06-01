import torch
import matplotlib.pyplot as plt


# data
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data(tensor), shape=[100, 1]
y = x.pow(2) + 0.1 * torch.rand(x.size())  # noisy y data(tensor), shape=[100,1]


def save():
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss()

    for i in range(100):
        prediction = net1(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # plot
    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

    # method 1: save entire network
    torch.save(net1, 'net.pkl')

    # method 2: save only the parameters
    torch.save(net1.state_dict(), 'net_params.pkl')


def reload_net():
    net2 = torch.load('net.pkl')
    prediction = net2(x)

    # plot
    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)


def reload_params():
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    net3.load_state_dict(torch.load('net_params.pkl'))
    prediction = net3(x)

    # plot
    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    plt.show()


# save net1
save()

# restore entire net (may slow)
reload_net()

# restore only the net parameters
reload_params()
