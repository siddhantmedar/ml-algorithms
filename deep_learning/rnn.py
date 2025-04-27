import numpy as np

np.random.seed(313)

# parameters
input_dim = 2  # scalar
hidden_dim = 5  # scalar
timestamps = 10  # scalar
output_dim = 2  # scalar

learning_rate = 0.001  # scalar
epochs = 100  # scalar

# data
x = np.random.randn(timestamps, input_dim)  # [timestamps, input_dim]
y_true = np.array(
    [np.random.randint(0, 2, output_dim) for _ in range(timestamps)]
)  # [timestamps, output_dim]

# weights
W_xh = np.random.randn(hidden_dim, input_dim)  # [hidden_dim, input_dim]
W_hh = np.random.randn(hidden_dim, hidden_dim)  # [hidden_dim, hidden_dim]
W_hy = np.random.randn(output_dim, hidden_dim)  # [output_dim, hidden_dim]

# training loop
for epoch in range(epochs):  # scalar
    h_prev = np.zeros(
        hidden_dim,
    )  # [hidden_dim] - prev hidden state

    # structures - per epoch
    hidden_states = []  # Will store tensors of shape [hidden_dim]
    outputs = []  # Will store tensors of shape [output_dim]
    loss = []  # Will store scalars

    dW_xh = np.zeros_like(W_xh)  # [hidden_dim, input_dim]
    dW_hh = np.zeros_like(W_hh)  # [hidden_dim, hidden_dim]
    dW_hy = np.zeros_like(W_hy)  # [output_dim, hidden_dim]

    # forward pass
    for t in range(timestamps):  # scalar
        # x[t] is [input_dim], h_prev is [hidden_dim]
        # W_xh @ x[t] = [hidden_dim, input_dim] @ [input_dim] = [hidden_dim]
        # W_hh @ h_prev = [hidden_dim, hidden_dim] @ [hidden_dim] = [hidden_dim]
        # The addition is [hidden_dim] + [hidden_dim] = [hidden_dim]
        h_t = np.tanh(W_xh @ x[t] + W_hh @ h_prev)  # [hidden_dim]

        # W_hy @ h_t = [output_dim, hidden_dim] @ [hidden_dim] = [output_dim]
        y_t = W_hy @ h_t  # [output_dim]

        # store info
        # y_t - y_true[t] = [output_dim] - [output_dim] = [output_dim]
        # ((y_t - y_true[t]) ** 2) = [output_dim]
        # np.sum(...) = scalar
        loss.append(0.5 * np.sum((y_t - y_true[t]) ** 2))  # scalar
        hidden_states.append(h_t)  # Appending [hidden_dim]
        outputs.append(y_t)  # Appending [output_dim]

        h_prev = h_t  # [hidden_dim]

    # backward pass
    for t in list(reversed(range(timestamps))):  # scalar
        dy = outputs[t] - y_true[t]  # [output_dim] - [output_dim] = [output_dim]

        # W_hy.T @ dy = [hidden_dim, output_dim] @ [output_dim] = [hidden_dim]
        dh = W_hy.T @ dy  # [hidden_dim]

        # np.outer(dy, hidden_states[t]) = [output_dim] ⊗ [hidden_dim] = [output_dim, hidden_dim]
        dW_hy += np.outer(dy, hidden_states[t])  # [output_dim, hidden_dim]

        # 1 - hidden_states[t] ** 2 = 1 - [hidden_dim]**2 = [hidden_dim]
        tanh_grad = 1 - hidden_states[t] ** 2  # [hidden_dim]

        # dh * tanh_grad = [hidden_dim] * [hidden_dim] = [hidden_dim] (elementwise)
        dz = dh * tanh_grad  # [hidden_dim]

        # np.outer(dz, x[t]) = [hidden_dim] ⊗ [input_dim] = [hidden_dim, input_dim]
        dW_xh += np.outer(dz, x[t])  # [hidden_dim, input_dim]

        if t > 0:
            # np.outer(dz, hidden_states[t-1]) = [hidden_dim] ⊗ [hidden_dim] = [hidden_dim, hidden_dim]
            dW_hh += np.outer(dz, hidden_states[t - 1])  # [hidden_dim, hidden_dim]
        else:
            # np.zeros_like(h_prev) = [hidden_dim]
            # np.outer(dz, np.zeros_like(h_prev)) = [hidden_dim] ⊗ [hidden_dim] = [hidden_dim, hidden_dim]
            dW_hh += np.outer(dz, np.zeros_like(h_prev))  # [hidden_dim, hidden_dim]

    # update parameters
    # W_xh and dW_xh both [hidden_dim, input_dim]
    W_xh -= learning_rate * dW_xh  # [hidden_dim, input_dim]

    # W_hh and dW_hh both [hidden_dim, hidden_dim]
    W_hh -= learning_rate * dW_hh  # [hidden_dim, hidden_dim]

    # W_hy and dW_hy both [output_dim, hidden_dim]
    W_hy -= learning_rate * dW_hy  # [output_dim, hidden_dim]

    if epoch % 10 == 0:
        print(f"Epoch: {epoch+1}/{epochs} | Average Train Loss: {np.mean(loss):.4f}")
        print(
            f"dW_xh norm: {np.linalg.norm(dW_xh):.4f}, dW_hh norm: {np.linalg.norm(dW_hh):.4f}, dW_hy norm: {np.linalg.norm(dW_hy):.4f}"
        )
        print("-" * 100)
