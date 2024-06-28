import torch
import torch.nn.functional as F

# Check for CUDA device and set as default if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Setting the seed for reproducibility
torch.manual_seed(42)

# Initialize matrices
m_y = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.0], [0.0, 0.5]]], device=device)
m_u = torch.tensor(
    [[[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.0], [0.0, 0.5]], [[0.25, 0.0], [0.0, 0.25]]],
    device=device,
)
y = torch.tensor(
    [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]], device=device
)
u = torch.tensor(
    [[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]], device=device
)

# Compute manually for t=0, t=1, t=2
# For t=0
u_0_m_u_0 = torch.einsum("sd,di->si", u[:, 0, :], m_u[0])
manual_t_0 = u_0_m_u_0

# For t=1
y_0_m_y_0 = torch.einsum("sd,di->si", y[:, 0, :], m_y[0])
u_1_m_u_0 = torch.einsum("sd,di->si", u[:, 1, :], m_u[0])
u_0_m_u_1 = torch.einsum("sd,di->si", u[:, 0, :], m_u[1])
manual_t_1 = y_0_m_y_0 + u_1_m_u_0 + u_0_m_u_1

# For t=2
y_1_m_y_0 = torch.einsum("sd,di->si", y[:, 1, :], m_y[0])
y_0_m_y_1 = torch.einsum("sd,di->si", y[:, 0, :], m_y[1])
u_2_m_u_0 = torch.einsum("sd,di->si", u[:, 2, :], m_u[0])
u_1_m_u_1 = torch.einsum("sd,di->si", u[:, 1, :], m_u[1])
u_0_m_u_2 = torch.einsum("sd,di->si", u[:, 0, :], m_u[2])
manual_t_2 = y_1_m_y_0 + y_0_m_y_1 + u_2_m_u_0 + u_1_m_u_1 + u_0_m_u_2


def shift(u: torch.Tensor, k: int = 1) -> torch.Tensor:
    """
    Shift time axis by k steps to align u_{t-k} with u_t.
    """
    if k == 0:
        return u  # No shift needed
    bsz, sl, d = u.shape
    padding = torch.zeros(bsz, k, d, device=u.device)
    if k < sl:
        shifted = torch.cat([padding, u[:, :-k]], dim=1)
    else:
        shifted = padding[:, :sl]
    return shifted


def compute_ar(
    y: torch.Tensor, u: torch.Tensor, m_y: torch.Tensor, m_u: torch.Tensor, k_y: int
) -> torch.Tensor:
    """
    Compute the autoregressive component based on past y and future u inputs.

    Args:
        y (torch.Tensor): A tensor of shape [bsz, sl, d_out].
        u (torch.Tensor): A tensor of shape [bsz, sl, d_in].
        m_y (torch.Tensor): A matrix of shape [k_y, d_out, d_out] that acts as windowed
            transition matrix for the linear dynamical system evolving y_t.
        m_u (torch.Tensor): A matrix of shape [k_u, d_out, d_in] that acts as windowed
            transition matrix for the linear dynamical system evolving u_t.
        k_y (int): The number of lags to use for the autoregressive component.

    Returns:
        torch.Tensor: A tensor of shape [bsz, sl, d_out].
    """
    bsz, sl, d_out = y.shape
    ar_component = torch.zeros(bsz, sl, d_out, device=y.device)

    print(f"compute_ar: Input shapes - y: {y.shape}, u: {u.shape}, m_y: {m_y.shape}, m_u: {m_u.shape}")

    for t in range(sl):
        print(f"\nTime step t={t}")
        # y component
        for i in range(1, min(t + 1, k_y) + 1):
            y_shifted = shift(y, i)
            y_contrib = torch.einsum("bd,od->bo", y_shifted[:, t, :], m_y[i - 1])
            print(f"  y contrib (i={i}): {y_contrib}")
            ar_component[:, t, :] += y_contrib

        # u component
        for i in range(min(t + 1, k_y + 1)):
            u_shifted = shift(u, i)
            u_contrib = torch.einsum("bd,di->bi", u_shifted[:, t, :], m_u[i])
            print(f"  u contrib (i={i}): {u_contrib}")
            ar_component[:, t, :] += u_contrib

        print(f"  Total for t={t}: {ar_component[:, t, :]}")

    return ar_component


def compute_ar_optimized(
    y: torch.Tensor, u: torch.Tensor, m_y: torch.Tensor, m_u: torch.Tensor, k_y: int
) -> torch.Tensor:
    """
    Compute the autoregressive component based on past y and future u inputs using vectorized operations.
    """
    """
    Compute the autoregressive component based on past y and future u inputs.

    Args:
        y (torch.Tensor): A tensor of shape [bsz, sl, d_out].
        u (torch.Tensor): A tensor of shape [bsz, sl, d_in].
        m_y (torch.Tensor): A matrix of shape [k_y, d_out, d_out] that acts as windowed
            transition matrix for the linear dynamical system evolving y_t.
        m_u (torch.Tensor): A matrix of shape [k_u, d_out, d_in] that acts as windowed
            transition matrix for the linear dynamical system evolving u_t.
        k_y (int): The number of lags to use for the autoregressive component.

    Returns:
        torch.Tensor: A tensor of shape [bsz, sl, d_out].
    """
    bsz, sl, d_out = y.shape
    ar_component = torch.zeros(bsz, sl, d_out, device=y.device)

    for i in range(1, k_y + 1):
        y_shifted = shift(y, i)
        ar_component += torch.einsum("btd,od->bto", y_shifted[:, :sl], m_y[i - 1])

    for i in range(k_y + 1):
        u_shifted = shift(u, i)
        ar_component += torch.einsum("btd,di->bti", u_shifted[:, :sl], m_u[i])

    return ar_component


# Initialize matrices
m_y = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.0], [0.0, 0.5]]], device=device)
m_u = torch.tensor(
    [[[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.0], [0.0, 0.5]], [[0.25, 0.0], [0.0, 0.25]]],
    device=device,
)
y = torch.tensor(
    [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]], device=device
)
u = torch.tensor(
    [[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]], device=device
)

# Test the function
k_y = 2  # max lag for y
try:
    ar_output = compute_ar_optimized(y, u, m_y, m_u, k_y)
    print("Shape of ar_output:", ar_output.shape)
    print("First few values of ar_output:")
    print(ar_output[0, :5, :])  # Print first 5 time steps of the first batch
except Exception as e:
    print(f"Error occurred: {str(e)}")
