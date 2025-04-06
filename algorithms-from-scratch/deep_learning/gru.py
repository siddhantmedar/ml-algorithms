import numpy as np

np.random.seed(313)


# parameters
input_dim = 2  # scalar
hidden_dim = 5  # scalar
timestamps = 3  # scalar
output_dim = 2  # scalar

learning_rate = 0.01  # scalar
epochs = 100  # scalar

# update gate
W_z = np.random.randn(hidden_dim, input_dim)
U_z = np.random.randn(hidden_dim, hidden_dim)

# reset gate
W_r = np.random.randn(hidden_dim, input_dim)
U_r = np.random.randn(hidden_dim, hidden_dim)

# candidate value
W_h = np.random.randn(hidden_dim, input_dim)
U_h = np.random.randn(hidden_dim, hidden_dim)

# output
W_hy = np.random.randn(output_dim, hidden_dim)  # [output_dim, hidden_dim]

# bias terms
b_z = np.zeros(hidden_dim)  # [hidden_dim]
b_r = np.zeros(hidden_dim)  # [hidden_dim]
b_h = np.zeros(hidden_dim)  # [hidden_dim]
b_hy = np.zeros(output_dim)  # [hidden_dim]

# data
x = np.random.randn(timestamps, input_dim)  # [timestamps, input_dim]
y_true = np.array(
    [np.random.randint(0, 2, output_dim) for _ in range(timestamps)]
)  # [timestamps, output_dim]


# activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # same shape as input x


for epoch in range(epochs):  # scalar

    # structures - per epoch
    h_prev = np.zeros(hidden_dim)  # [hidden_dim]
    hidden_states = []  # will contain tensors of shape [hidden_dim]
    outputs = []  # will contain tensors of shape [output_dim]
    loss = []  # will contain scalar values

    z_s, r_s, g_s = [], [], []

    # forward pass
    for t in range(timestamps):  # scalar
        z_t = sigmoid(W_z @ x[t] + U_z @ h_prev + b_z)  # [hidden_dim]
        r_t = sigmoid(W_r @ x[t] + U_r @ h_prev + b_r)  # [hidden_dim]

        # candidate value
        g_t = np.tanh(W_h @ x[t] + U_h @ (r_t * h_prev) + b_h)  # [hidden_dim]

        h_t = (1 - z_t) * h_prev + z_t * g_t  # [hidden_dim]

        y_t = W_hy @ h_t + b_hy  # [output_dim]

        hidden_states.append(h_t)  # append [hidden_dim]
        outputs.append(y_t)  # append [output_dim]
        loss.append(0.5 * np.sum((y_t - y_true[t]) ** 2))  # append scalar

        z_s.append(z_t)
        r_s.append(r_t)
        g_s.append(g_t)

        h_prev = h_t  # [hidden_dim]

    dW_z = np.zeros_like(W_z)  # [hidden_dim, input_dim]
    dU_z = np.zeros_like(U_z)  # [hidden_dim, hidden_dim]
    db_z = np.zeros_like(b_z)  # [hidden_dim]

    dW_r = np.zeros_like(W_r)  # [hidden_dim, input_dim]
    dU_r = np.zeros_like(U_r)  # [hidden_dim, hidden_dim]
    db_r = np.zeros_like(b_r)  # [hidden_dim]

    dW_h = np.zeros_like(W_h)  # [hidden_dim, input_dim]
    dU_h = np.zeros_like(U_h)  # [hidden_dim, hidden_dim]
    db_h = np.zeros_like(b_h)  # [hidden_dim]

    dW_hy = np.zeros_like(W_hy)  # [hidden_dim, input_dim]
    db_hy = np.zeros_like(b_hy)  # [output_dim]

    dh_prev = np.zeros_like(h_prev)  # [hidden_dim]

    # backward pass
    for t in list(reversed(range(timestamps))):  # scalar

        z_t = z_s[t]
        r_t = r_s[t]
        g_t = g_s[t]
        h_t = hidden_states[t]
        h_prev = hidden_states[t - 1] if t > 0 else np.zeros_like(h_prev)

        # from output layer
        dy = outputs[t] - y_true[t]  # [output_dim]
        dW_hy += np.outer(dy, h_t)  # [output_dim, hidden_dim]
        db_hy += dy
        dh = W_hy.T @ dy + dh_prev  # [hidden_dim]

        # from hidden layer
        dg_t = dh * z_t  # [hidden_dim]
        dz_t = dh * (g_t - h_prev)  # [hidden_dim]
        dh_prev += dh * (1 - z_t)  # [hidden_dim] -> (h_t-1(1))

        # candidate state (g_t)
        dpreact_g_t = dg_t * (1 - g_t**2)  # [hidden_dim]
        dW_h += np.outer(dpreact_g_t, x[t])  # [hidden_dim,input_dim]
        dU_h += np.outer(dpreact_g_t, (r_t * h_prev))  # [hidden_dim,hidden_dim]
        db_h += dpreact_g_t  # [hidden_dim]
        dr_t = (U_h @ dpreact_g_t) * h_prev  # [hidden_dim]
        dh_prev += (dpreact_g_t @ U_h) * h_prev  # [hidden_dim] -> (h_t-1(2))

        # reset gate
        dpreact_r = dr_t * r_t * (1 - r_t)  # [hidden_dim]
        dW_r += np.outer(dpreact_r, x[t])  # [hidden_dim,input_dim]
        dU_r += np.outer(dpreact_r, h_prev)  # [hidden_dim,hidden_dim]
        db_r += dpreact_r  # [hidden_dim]
        dh_prev += U_r @ dpreact_r  # [hidden_dim] -> (h_t-1(3))

        # update gate
        dpreact_z = dz_t * z_t * (1 - z_t)  # [hidden_dim]
        dW_z += np.outer(dpreact_z, x[t])  # [hidden_dim,input_dim]
        dU_z += np.outer(dpreact_z, h_prev)  # [hidden_dim,hidden_dim]
        db_z += dpreact_z
        dh_prev += U_z @ dpreact_z  # [hidden_dim] -> (h_t-1(4))

    # update parameters
    W_z -= learning_rate * dW_z  # [hidden_dim, input_dim]
    U_z -= learning_rate * dU_z  # [hidden_dim, hidden_dim]
    b_z -= learning_rate * db_z  # [hidden_dim]

    W_r -= learning_rate * dW_r  # [hidden_dim, input_dim]
    U_r -= learning_rate * dU_r  # [hidden_dim, hidden_dim]
    b_r -= learning_rate * db_r  # [hidden_dim]

    W_h -= learning_rate * dW_h  # [hidden_dim, input_dim]
    U_h -= learning_rate * dU_h  # [hidden_dim, hidden_dim]
    b_h -= learning_rate * db_h  # [hidden_dim]

    W_hy -= learning_rate * dW_hy  # [output_dim, hidden_dim]
    b_hy -= learning_rate * db_hy  # [output_dim]

    if epoch % 10 == 0:
        print(f"Epoch: {epoch+1}/{epochs} | Average Train Loss: {np.mean(loss):.4f}")
        print(
            f"dW_z norm: {np.linalg.norm(dW_z):.4f}, dU_z norm: {np.linalg.norm(dU_z):.4f}, db_z norm: {np.linalg.norm(db_z):.4f}"
        )
        print(
            f"dW_r norm: {np.linalg.norm(dW_r):.4f}, dU_r norm: {np.linalg.norm(dU_r):.4f}, db_r norm: {np.linalg.norm(db_r):.4f}"
        )
        print(
            f"dW_h norm: {np.linalg.norm(dW_h):.4f}, dU_h norm: {np.linalg.norm(dU_h):.4f}, db_h norm: {np.linalg.norm(db_h):.4f}"
        )
        print(
            f"dW_hy norm: {np.linalg.norm(dW_hy):.4f}, db_hy norm: {np.linalg.norm(db_hy):.4f}"
        )
        print(
            f"b_z norm: {np.linalg.norm(b_z):.4f}, b_r norm: {np.linalg.norm(b_r):.4f}, b_h norm: {np.linalg.norm(b_h):.4f}, b_hy norm: {np.linalg.norm(b_hy):.4f}"
        )
        print("-" * 100)
