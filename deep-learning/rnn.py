import numpy as np
np.random.seed(313)

# parameters
input_dim = 2
hidden_dim = 5
timestamps = 3
out_dim = 2

x = np.random.randn(timestamps,input_dim)
y_true = np.array([np.random.randint(0,2,out_dim) for _ in range(timestamps)])

# prev hidden state
h_prev = np.zeros(hidden_dim,)

# weights
W_xh = np.random.randn(hidden_dim,input_dim)
W_hh = np.random.randn(hidden_dim,hidden_dim)
W_hy = np.random.randn(out_dim,hidden_dim)

hidden_states = []
outputs = []
loss = []

# forward pass
for t in range(timestamps):    
    h_t = np.tanh(W_xh @ x[t] + W_hh @ h_prev)  # h_t = np.tanh(np.dot(W_xh,x[t])+ np.dot(W_hh,h_prev))
    y_t = W_hy @ h_t                            # y_t = np.dot(W_hy,h_t)

    # compute loss
    loss.append(np.round(0.5*np.sum((y_t-y_true[t])**2),2))

    hidden_states.append(h_t)
    outputs.append(y_t)

    print(f"At timestamp {t}: h_t: {h_t} | y_t: {y_t} | Loss: {loss[-1]}")

    h_prev = h_t

print(f"Loss (all timestamps): {loss}")