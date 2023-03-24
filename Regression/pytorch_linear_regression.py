import torch
from torch.autograd import Variable

x = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))

class Model(torch.nn.Module):
    def __init__(self):
        """Model class constructor"""
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):

        pred = self.linear(x)
        return pred

model = Model()

loss_function = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for epoch in range(500):

    preds = model(x)
    loss = loss_function(preds, y)
    if epoch % 100 == 0:
        print(epoch)
        print(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

predict = model.forward(torch.Tensor([[150.0]]))
print(predict)
