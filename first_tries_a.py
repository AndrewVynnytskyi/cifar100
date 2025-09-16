# %%
import torch
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%

# Initializing tensor

x = torch.empty(3)
print(x)
y = torch.empty(5, 2)
print(y)
z = torch.randn(5, 2, 3)
print(z)

# %%
print(z.size())
print(z.shape)

print(z.dtype)

d = torch.randn(4, 2, dtype=float)

print(d.dtype)

# %%

x = torch.randn(4, 4)
print(x)

y = x.view(16)
print(y)

# %%

x = torch.randn(2, 4, requires_grad=True)
y = torch.ones(2, 4, requires_grad=True)

z = x ** 2 + y
print(y.grad_fn)
z = z - 2
print(z.grad_fn)
z = z.mean()

z.backward()
print(x.grad)  # dz/dx
print(y.grad)  # dz/dy

# %%

x = torch.tensor([[3., 7.],
                  [5., 2.],
                  [0., 9.],
                  [8., 6.],
                  [2., 4.],
                  [7., 0.],
                  [10., 10.],
                  [1., 3.],
                  [4., 8.],
                  [6., 5.],
                  [9., 1.],
                  [2., 0.],
                  [0., 5.],
                  [8., 3.],
                  [1., 9.],
                  [5., 10.],
                  [4., 0.],
                  [7., 6.],
                  [3., 2.],
                  [10., 0.]
                  ])

y = torch.tensor([34., 8., 46., 18., 22., -4., 50., 20., 38., 18., 6., 6., 30., 20.,
                  44., 40., 2., 26., 20., -10.
                  ], dtype=torch.float32)

bias = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

weights = torch.tensor([0.0, 0.0], dtype=torch.float32, requires_grad=True)

def forward(_x) -> torch.Tensor:
    return (weights * _x).sum(dim=1) + bias

#MSE - loss
def loss(y_actual: torch.Tensor, y_pred: torch.Tensor):
    return ((y_actual - y_pred)**2).mean()

# %%

learning_rate = 0.01
epochs = 10000

for epoch in range(epochs):

    y_pred = forward(x)

    l = loss(y, y_pred)

    l.backward()

    with torch.no_grad():
        weights -= learning_rate * weights.grad
        bias -= learning_rate *bias.grad

    weights.grad.zero_()
    bias.grad.zero_()

    if epoch % 100 == 0:
        print(f"Epoch {epoch} loss: {l}, weights: {weights.data}, bias: {bias.item()}")


print('test:', forward(torch.tensor([
    [0., 10.],  # 0 кави, 10 спорту
    [5., 5.],   # 5 кави, 5 спорту
    [2., 7.]    # 2 кави, 7 спорту
])))

# %%

x = torch.tensor([[3., 7.],
                  [5., 2.],
                  [0., 9.],
                  [8., 6.],
                  [2., 4.],
                  [7., 0.],
                  [10., 10.],
                  [1., 3.],
                  [4., 8.],
                  [6., 5.],
                  [9., 1.],
                  [2., 0.],
                  [0., 5.],
                  [8., 3.],
                  [1., 9.],
                  [5., 10.],
                  [4., 0.],
                  [7., 6.],
                  [3., 2.],
                  [10., 0.]
                  ])

y = torch.tensor([34., 8., 46., 18., 22., -4., 50., 20., 38., 18., 6., 6., 30., 20.,
                  44., 40., 2., 26., 20., -10.
                  ], dtype=torch.float32)

y = y.view(-1, 1)# pytotch погано бродкастить не забувати переприсвоювати

n_samples, n_features = x.shape
print(n_samples, n_features)

# %%

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, _x):
        return self.lin(_x)


input_size, output_size = n_features, 1

model = LinearRegression(input_size, output_size)

learning_rate = 0.1
epochs = 100000

loss = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    y_pred = model.forward(x)

    l = loss(y, y_pred)

    l.backward()

    optimizer.step()

    optimizer.zero_grad()

    if epoch % 100 == 0:
        w, b = model.parameters()
        print(f"Epoch {epoch} loss: {l}, weights: {w[0].data}, bias: {b.item()}")


print('test:', model.forward(torch.tensor([
    [0., 10.],  # 0 кави, 10 спорту
    [5., 5.],   # 5 кави, 5 спорту
    [2., 7.]    # 2 кави, 7 спорту
])))

