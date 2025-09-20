# %%
import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

z = x**2 + y
print(y.grad_fn)
z = z - 2
print(z.grad_fn)
z = z.mean()

z.backward()
print(x.grad)  # dz/dx
print(y.grad)  # dz/dy

# %%

x = torch.tensor(
    [
        [3.0, 7.0],
        [5.0, 2.0],
        [0.0, 9.0],
        [8.0, 6.0],
        [2.0, 4.0],
        [7.0, 0.0],
        [10.0, 10.0],
        [1.0, 3.0],
        [4.0, 8.0],
        [6.0, 5.0],
        [9.0, 1.0],
        [2.0, 0.0],
        [0.0, 5.0],
        [8.0, 3.0],
        [1.0, 9.0],
        [5.0, 10.0],
        [4.0, 0.0],
        [7.0, 6.0],
        [3.0, 2.0],
        [10.0, 0.0],
    ]
)

y = torch.tensor(
    [
        34.0,
        8.0,
        46.0,
        18.0,
        22.0,
        -4.0,
        50.0,
        20.0,
        38.0,
        18.0,
        6.0,
        6.0,
        30.0,
        20.0,
        44.0,
        40.0,
        2.0,
        26.0,
        20.0,
        -10.0,
    ],
    dtype=torch.float32,
)

bias = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

weights = torch.tensor([0.0, 0.0], dtype=torch.float32, requires_grad=True)


def forward(_x) -> torch.Tensor:
    return (weights * _x).sum(dim=1) + bias


# MSE - loss
def loss(y_actual: torch.Tensor, y_pred: torch.Tensor):
    return ((y_actual - y_pred) ** 2).mean()


# %%

learning_rate = 0.01
epochs = 10000

for epoch in range(epochs):
    y_pred = forward(x)

    loss_value = loss(y, y_pred)
    loss_value.backward()

    with torch.no_grad():
        weights -= learning_rate * weights.grad
        bias -= learning_rate * bias.grad

    weights.grad.zero_()
    bias.grad.zero_()

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch}, "
            f"loss: {loss_value}, "
            f"weights: {weights.data}, "
            f"bias: {bias.item()}"
        )

print(
    "test:",
    forward(
        torch.tensor(
            [
                [0.0, 10.0],  # 0 кави, 10 спорту
                [5.0, 5.0],  # 5 кави, 5 спорту
                [2.0, 7.0],  # 2 кави, 7 спорту
            ]
        )
    ),
)

# %%

x = torch.tensor(
    [
        [3.0, 7.0],
        [5.0, 2.0],
        [0.0, 9.0],
        [8.0, 6.0],
        [2.0, 4.0],
        [7.0, 0.0],
        [10.0, 10.0],
        [1.0, 3.0],
        [4.0, 8.0],
        [6.0, 5.0],
        [9.0, 1.0],
        [2.0, 0.0],
        [0.0, 5.0],
        [8.0, 3.0],
        [1.0, 9.0],
        [5.0, 10.0],
        [4.0, 0.0],
        [7.0, 6.0],
        [3.0, 2.0],
        [10.0, 0.0],
    ]
)

y = torch.tensor(
    [
        34.0,
        8.0,
        46.0,
        18.0,
        22.0,
        -4.0,
        50.0,
        20.0,
        38.0,
        18.0,
        6.0,
        6.0,
        30.0,
        20.0,
        44.0,
        40.0,
        2.0,
        26.0,
        20.0,
        -10.0,
    ],
    dtype=torch.float32,
)

y = y.view(-1, 1)  # pytotch погано бродкастить не забувати переприсвоювати

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

    loss_value = loss(y, y_pred)
    loss_value.backward()

    optimizer.step()
    optimizer.zero_grad()

    if epoch % 100 == 0:
        w, b = model.parameters()
        print(
            f"Epoch {epoch}, "
            f"loss: {loss_value}, "
            f"weights: {w[0].data}, "
            f"bias: {b.item()}"
        )


print(
    "test:",
    model.forward(
        torch.tensor(
            [
                [0.0, 10.0],  # 0 кави, 10 спорту
                [5.0, 5.0],  # 5 кави, 5 спорту
                [2.0, 7.0],  # 2 кави, 7 спорту
            ]
        )
    ),
)
