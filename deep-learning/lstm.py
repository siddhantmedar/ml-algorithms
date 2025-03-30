import numpy as np

np.random.seed(313)


# parameters
input_dim = 2  # scalar
hidden_dim = 5  # scalar
timestamps = 3  # scalar
output_dim = 2  # scalar

learning_rate = 0.01  # scalar
epochs = 100  # scalar

ct_prev = np.zeros(hidden_dim)  # [hidden_dim]

# forget
W_f = np.random.randn(hidden_dim, input_dim)  # [hidden_dim, input_dim]
U_f = np.random.randn(hidden_dim, hidden_dim)  # [hidden_dim, hidden_dim]
# input
W_i = np.random.randn(hidden_dim, input_dim)  # [hidden_dim, input_dim]
U_i = np.random.randn(hidden_dim, hidden_dim)  # [hidden_dim, hidden_dim]
# candidate
W_c = np.random.randn(hidden_dim, input_dim)  # [hidden_dim, input_dim]
U_c = np.random.randn(hidden_dim, hidden_dim)  # [hidden_dim, hidden_dim]
# output
W_o = np.random.randn(hidden_dim, input_dim)  # [hidden_dim, input_dim]
U_o = np.random.randn(hidden_dim, hidden_dim)  # [hidden_dim, hidden_dim]

W_hy = np.random.randn(output_dim, hidden_dim)  # [output_dim, hidden_dim]

# bias terms
b_f = np.zeros(hidden_dim)  # [hidden_dim]
b_i = np.zeros(hidden_dim)  # [hidden_dim]
b_c = np.zeros(hidden_dim)  # [hidden_dim]
b_o = np.zeros(hidden_dim)  # [hidden_dim]
b_hy = np.zeros(output_dim)  # [output_dim]

# data
x = np.random.randn(timestamps, input_dim)  # [timestamps, input_dim]
y_true = np.array([np.random.randint(0, 2, output_dim) for _ in range(timestamps)])  # [timestamps, output_dim]


# activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # same shape as input x


for epoch in range(epochs):  # scalar
    
    # structures - per epoch
    h_prev = np.zeros(hidden_dim)  # [hidden_dim]
    hidden_states = []  # will contain tensors of shape [hidden_dim]
    outputs = []  # will contain tensors of shape [output_dim]
    loss = []  # will contain scalar values

    f_s, i_s, o_s, g_s, c_s, h_s = [], [], [], [], [], []  # will contain tensors of shape [hidden_dim]

    # forward pass
    for t in range(timestamps):  # scalar
        f_t = sigmoid(W_f @ x[t] + U_f @ h_prev + b_f)  # [hidden_dim]
        i_t = sigmoid(W_i @ x[t] + U_i @ h_prev + b_i)  # [hidden_dim]
        o_t = sigmoid(W_o @ x[t] + U_o @ h_prev + b_o)  # [hidden_dim]

        g_t = np.tanh(W_c @ x[t] + U_c @ h_prev + b_c)  # [hidden_dim]

        c_t = f_t * ct_prev + i_t * g_t  # [hidden_dim]

        h_t = o_t * np.tanh(c_t)  # [hidden_dim]

        y_t = W_hy @ h_t + b_hy  # [output_dim]

        hidden_states.append(h_t)  # append [hidden_dim]
        outputs.append(y_t)  # append [output_dim]
        loss.append(0.5 * np.sum((y_t - y_true[t]) ** 2))  # append scalar

        f_s.append(f_t)  # append [hidden_dim]
        i_s.append(i_t)  # append [hidden_dim]
        o_s.append(o_t)  # append [hidden_dim]
        g_s.append(g_t)  # append [hidden_dim]
        c_s.append(c_t)  # append [hidden_dim]
        h_s.append(h_t)  # append [hidden_dim]

        h_prev = h_t  # [hidden_dim]
        ct_prev = c_t  # [hidden_dim]

    dW_f = np.zeros_like(W_f)  # [hidden_dim, input_dim]
    dU_f = np.zeros_like(U_f)  # [hidden_dim, hidden_dim]
    db_f = np.zeros_like(b_f)  # [hidden_dim]

    dW_i = np.zeros_like(W_i)  # [hidden_dim, input_dim]
    dU_i = np.zeros_like(U_i)  # [hidden_dim, hidden_dim]
    db_i = np.zeros_like(b_i)  # [hidden_dim]

    dW_c = np.zeros_like(W_c)  # [hidden_dim, input_dim]
    dU_c = np.zeros_like(U_c)  # [hidden_dim, hidden_dim]
    db_c = np.zeros_like(b_c)  # [hidden_dim]

    dW_o = np.zeros_like(W_o)  # [hidden_dim, input_dim]
    dU_o = np.zeros_like(U_o)  # [hidden_dim, hidden_dim]
    db_o = np.zeros_like(b_o)  # [hidden_dim]

    dW_hy = np.zeros_like(W_hy)  # [output_dim, hidden_dim]
    db_hy = np.zeros_like(b_hy)  # [output_dim]

    dh_prev = np.zeros_like(h_prev)  # [hidden_dim]
    dc_prev = np.zeros_like(ct_prev)  # [hidden_dim]

    # backward pass
    for t in list(reversed(range(timestamps))):  # scalar

        ct_prev = c_s[t - 1] if t > 0 else np.zeros(hidden_dim)  # [hidden_dim]
        h_prev = h_s[t - 1] if t > 0 else np.zeros(hidden_dim)  # [hidden_dim]

        f_t = f_s[t]  # [hidden_dim]
        i_t = i_s[t]  # [hidden_dim]
        o_t = o_s[t]  # [hidden_dim]
        g_t = g_s[t]  # [hidden_dim]
        c_t = c_s[t]  # [hidden_dim]
        h_t = h_s[t]  # [hidden_dim]

        dy = outputs[t] - y_true[t]  # [output_dim]
        dW_hy += np.outer(dy, h_t)  # [output_dim, hidden_dim]
        dh = W_hy.T @ dy + dh_prev  # [hidden_dim]
        do_t = dh * np.tanh(c_t)  # [hidden_dim]
        dc_t = dh * (1 - np.tanh(c_t) ** 2) * o_t + dc_prev  # [hidden_dim]

        # backprop post activation
        df_t = dc_t * ct_prev  # [hidden_dim]
        di_t = dc_t * g_t  # [hidden_dim]
        dg_t = dc_t * i_t  # [hidden_dim]
        dct_prev = dc_t * f_t  # [hidden_dim]

        # backprop pre activation (raw)
        dz_f = df_t * f_t * (1 - f_t)  # [hidden_dim]
        dz_i = di_t * i_t * (1 - i_t)  # [hidden_dim]
        dz_o = do_t * o_t * (1 - o_t)  # [hidden_dim]
        dz_g = dg_t * (1 - g_t**2)  # [hidden_dim]

        dW_f += np.outer(dz_f, x[t])  # [hidden_dim, input_dim]
        dU_f += np.outer(dz_f, h_prev)  # [hidden_dim, hidden_dim]
        db_f += dz_f  # [hidden_dim]

        dW_i += np.outer(dz_i, x[t])  # [hidden_dim, input_dim]
        dU_i += np.outer(dz_i, h_prev)  # [hidden_dim, hidden_dim]
        db_i += dz_i  # [hidden_dim]

        dW_o += np.outer(dz_o, x[t])  # [hidden_dim, input_dim]
        dU_o += np.outer(dz_o, h_prev)  # [hidden_dim, hidden_dim]
        db_o += dz_o  # [hidden_dim]

        dW_c += np.outer(dz_g, x[t])  # [hidden_dim, input_dim]
        dU_c += np.outer(dz_g, h_prev)  # [hidden_dim, hidden_dim]
        db_c += dz_g  # [hidden_dim]

        dh_prev = U_f @ dz_f + U_i @ dz_i + U_c @ dz_g + U_o @ dz_o  # [hidden_dim]
        dc_prev = dc_t * f_t  # [hidden_dim]

        db_hy += dy  # [output_dim]

    # update parameters
    W_f -= learning_rate * dW_f  # [hidden_dim, input_dim]
    U_f -= learning_rate * dU_f  # [hidden_dim, hidden_dim]
    b_f -= learning_rate * db_f  # [hidden_dim]

    W_i -= learning_rate * dW_i  # [hidden_dim, input_dim]
    U_i -= learning_rate * dU_i  # [hidden_dim, hidden_dim]
    b_i -= learning_rate * db_i  # [hidden_dim]

    W_c -= learning_rate * dW_c  # [hidden_dim, input_dim]
    U_c -= learning_rate * dU_c  # [hidden_dim, hidden_dim]
    b_c -= learning_rate * db_c  # [hidden_dim]

    W_o -= learning_rate * dW_o  # [hidden_dim, input_dim]
    U_o -= learning_rate * dU_o  # [hidden_dim, hidden_dim]
    b_o -= learning_rate * db_o  # [hidden_dim]

    W_hy -= learning_rate * dW_hy  # [output_dim, hidden_dim]
    b_hy -= learning_rate * db_hy  # [output_dim]

    if epoch % 10 == 0:
        print(f"Epoch: {epoch+1}/{epochs} | Average Train Loss: {np.mean(loss):.4f}")
        print(
            f"dW_f norm: {np.linalg.norm(dW_f):.4f}, dU_f norm: {np.linalg.norm(dU_f):.4f}, db_f norm: {np.linalg.norm(db_f):.4f}"
        )
        print(
            f"dW_i norm: {np.linalg.norm(dW_i):.4f}, dU_i norm: {np.linalg.norm(dU_i):.4f}, db_i norm: {np.linalg.norm(db_i):.4f}"
        )
        print(
            f"dW_c norm: {np.linalg.norm(dW_c):.4f}, dU_c norm: {np.linalg.norm(dU_c):.4f}, db_c norm: {np.linalg.norm(db_c):.4f}"
        )
        print(
            f"dW_o norm: {np.linalg.norm(dW_o):.4f}, dU_o norm: {np.linalg.norm(dU_o):.4f}, db_o norm: {np.linalg.norm(db_o):.4f}"
        )
        print(
            f"dW_hy norm: {np.linalg.norm(dW_hy):.4f}, db_hy norm: {np.linalg.norm(db_hy):.4f}"
        )
        print("-" * 100)

"""
Equations
f_t = σ(W_f @ x_t + U_f @ h_{t-1} + b_f)
i_t = σ(W_i @ x_t + U_i @ h_{t-1} + b_i)
g_t = tanh(W_c @ x_t + U_c @ h_{t-1} + b_c)
c_t = f_t * c_{t-1} + i_t * g_t
o_t = σ(W_o @ x_t + U_o @ h_{t-1} + b_o)
h_t = o_t * tanh(c_t)
y_t = W_hy @ h_t
L = 0.5 * ||y_t - y_true||² 
"""