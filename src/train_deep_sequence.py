import numpy as np
from sklearn.preprocessing import StandardScaler

class DemandLSTM:
    def __init__(self, input_size=7, hidden_size=64, lr=0.01, seed=42):
        rng = np.random.default_rng(seed)
        self.hidden_size = hidden_size
        self.lr = lr

        limit = np.sqrt(1.0 / input_size)
        self.params = {
            "Wx": rng.uniform(-limit, limit, (input_size, 4 * hidden_size)),
            "bx": np.zeros((1, 4 * hidden_size)),
            "c0": np.zeros((1, hidden_size)),
            "Wy": rng.uniform(-0.1, 0.1, (hidden_size, 1)),
            "by": np.zeros((1, 1)),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

    def forward(self, X):
        H = self.hidden_size
        Z = X @ self.params["Wx"] + self.params["bx"]
        f = self._sigmoid(Z[:, :H])
        i = self._sigmoid(Z[:, H:2 * H])
        g = np.tanh(Z[:, 2 * H:3 * H])
        o = self._sigmoid(Z[:, 3 * H:])

        c = f * self.params["c0"] + i * g
        tanh_c = np.tanh(c)
        h = o * tanh_c
        y_hat = h @ self.params["Wy"] + self.params["by"]

        cache = (X, f, i, g, o, tanh_c, h)
        return y_hat, cache

    def backward_and_step(self, y_hat, y_true, cache):
        X, f, i, g, o, tanh_c, h = cache
        B = X.shape[0]

        dy = (2.0 / B) * (y_hat - y_true)
        dWy = h.T @ dy
        dby = np.sum(dy, axis=0, keepdims=True)

        dh = dy @ self.params["Wy"].T
        do = dh * tanh_c
        dZo = do * o * (1.0 - o)

        dc = dh * o * (1.0 - tanh_c ** 2)
        dc0 = np.sum(dc * f, axis=0, keepdims=True)
        df = dc * self.params["c0"]
        dZf = df * f * (1.0 - f)

        di = dc * g
        dZi = di * i * (1.0 - i)

        dg = dc * i
        dZg = dg * (1.0 - g ** 2)

        dZ = np.hstack([dZf, dZi, dZg, dZo])
        dWx = X.T @ dZ
        dbx = np.sum(dZ, axis=0, keepdims=True)

        grads = {"Wx": dWx, "bx": dbx, "c0": dc0, "Wy": dWy, "by": dby}
        self.t += 1
        for k in self.params:
            self.m[k] = 0.9 * self.m[k] + 0.1 * grads[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * (grads[k] ** 2)
            m_hat = self.m[k] / (1.0 - 0.9 ** self.t)
            v_hat = self.v[k] / (1.0 - 0.999 ** self.t)
            self.params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)

def train_and_predict_lstm(X_train, y_train, X_test, epochs=150, batch_size=64):
    print("[*] Training Deep Sequence Architecture (LSTM Network)...")
    rng = np.random.default_rng(42)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))

    model = DemandLSTM(input_size=X_train.shape[1], hidden_size=64, lr=0.01)
    n_samples = X_train_scaled.shape[0]

    for _ in range(epochs):
        indices = rng.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            idx = indices[start:start + batch_size]
            y_hat, cache = model.forward(X_train_scaled[idx])
            model.backward_and_step(y_hat, y_train_scaled[idx], cache)

    test_preds_scaled, _ = model.forward(X_test_scaled)
    return scaler_y.inverse_transform(test_preds_scaled).flatten()