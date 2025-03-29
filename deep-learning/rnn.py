import numpy as np
np.random.seed(313)

# parameters
input_dim = 2
hidden_dim = 5
timestamps = 10
out_dim = 2

learning_rate = 0.001
epochs = 100

# data
x = np.random.randn(timestamps,input_dim)
y_true = np.array([np.random.randint(0,2,out_dim) for _ in range(timestamps)])

# weights
W_xh = np.random.randn(hidden_dim,input_dim)
W_hh = np.random.randn(hidden_dim,hidden_dim)
W_hy = np.random.randn(out_dim,hidden_dim)

# training loop
for epoch in range(epochs):
    h_prev = np.zeros(hidden_dim,) # prev hidden state

    # structures - per epoch
    hidden_states = []
    outputs = []
    loss = []

    dW_xh = np.zeros_like(W_xh)
    dW_hh = np.zeros_like(W_hh)
    dW_hy = np.zeros_like(W_hy)

    # forward pass
    for t in range(timestamps):    
        h_t = np.tanh(W_xh @ x[t] + W_hh @ h_prev)
        y_t = W_hy @ h_t
        
        # store info
        loss.append(0.5*np.sum((y_t-y_true[t])**2)) # compute loss
        hidden_states.append(h_t)
        outputs.append(y_t)

        # print(f"At timestamp {t}: h_t: {h_t} | y_t: {y_t} | Loss: {loss[-1]}")

        h_prev = h_t


    # backward pass
    for t in list(reversed(range(timestamps))):
        dy = outputs[t]-y_true[t]
        dh = W_hy.T @ dy
        dW_hy += np.outer(dy,hidden_states[t])
        tanh_grad = 1-hidden_states[t]**2
        dz = dh * tanh_grad
        dW_xh += np.outer(dz,x[t])
        dW_hh += np.outer(dz,hidden_states[t-1]) if t>0 else np.outer(dz,np.zeros_like(h_prev))


    # update parameters
    W_xh -= learning_rate * dW_xh
    W_hh -= learning_rate * dW_hh
    W_hy -= learning_rate * dW_hy

    if epoch%10==0:
        print(f"Epoch: {epoch+1}/{epochs} | Average Train Loss: {np.mean(loss):.4f}")
        print(f"dW_xh norm: {np.linalg.norm(dW_xh):.4f}, dW_hh norm: {np.linalg.norm(dW_hh):.4f}, dW_hy norm: {np.linalg.norm(dW_hy):.4f}")
        print("-"*100)



