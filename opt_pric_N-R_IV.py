import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm

# --- 1. Define Missing Financial Functions ---

def d1(S, K, T, r, sigma):
    return (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def bs(flag, S, K, T, r, sigma):
    """Calculate Black-Scholes Price"""
    if sigma <= 0: return 0.0 # Safety for negative vol
    
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)
    
    if flag == 'c':
        return S * norm.cdf(d_1) - K * np.exp(-r * T) * norm.cdf(d_2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d_2) - S * norm.cdf(-d_1)

def vega(flag, S, K, T, r, sigma):
    """Calculate Vega (derivative of price wrt volatility)"""
    if sigma <= 0: return 0.0
    d_1 = d1(S, K, T, r, sigma)
    return S * np.sqrt(T) * norm.pdf(d_1)

# --- 2. Implied Volatility Function with Animation Data ---

def implied_vol_solver(S0, K, T, r, market_price, flag='c', tol=0.00001):
    """
    Compute IV and return history for animation.
    Uses Newton-Raphson method.
    """
    max_iter = 300 
    vol_old = 0.50 # Start with a higher guess to make the animation more visible
    
    # Store steps for plotting: [x_start, x_end], [y_start, y_end]
    x_vals = []
    y_vals = []

    for k in range(max_iter):
        bs_price = bs(flag, S0, K, T, r, vol_old)
        vega_val = vega(flag, S0, K, T, r, vol_old)
        
        # Avoid division by zero
        if abs(vega_val) < 1e-8:
            break
            
        diff = bs_price - market_price
        
        # Newton-Raphson Step: x_new = x_old - f(x) / f'(x)
        vol_new = vol_old - diff / vega_val

        # 1. Visualize finding the point on the curve (Vertical move)
        x_vals.append([vol_old*100, vol_old*100]) # X stays same
        y_vals.append([market_price, bs_price])   # Y moves from Market to Model Price

        # 2. Visualize the tangent slope correction (Slide down tangent)
        # Note: We approximate the tangent line for visualization
        x_vals.append([vol_old*100, vol_new*100])
        y_vals.append([bs_price, market_price]) # Target Y is market price

        if abs(diff) < tol:
            break

        vol_old = vol_new

    return vol_old, x_vals, y_vals

# --- 3. Setup Data and Plotting ---

# Parameters
S0, K, T, r = 30, 28, 0.2, 0.025
market_price = 3.97

# 1. Generate the static curve for the background
vols_for_plot = []
prices_for_plot = []
for sigma_pct in range(1, 200):
    sigma = sigma_pct / 100.0
    p = bs('c', S0, K, T, r, sigma)
    vols_for_plot.append(sigma_pct)
    prices_for_plot.append(p)

# 2. Run the Solver
iv_est, x_vals, y_vals = implied_vol_solver(S0, K, T, r, market_price, flag='c')

print(f"Market Price: ${market_price}")
print(f"Calculated Implied Volatility: {iv_est*100:.2f}%")

# --- 4. Animation Setup ---

fig, ax = plt.subplots(figsize=(8, 6))
plt.title('Newton-Raphson Convergence for IV')
plt.ylabel('Call Price ($)')
plt.xlabel('Implied Volatility (%)')
plt.grid(True, alpha=0.3)

# Static Plots
ax.plot(vols_for_plot, prices_for_plot, label='Black-Scholes Curve', color='navy')
ax.axhline(y=market_price, color='gray', linestyle='--', label='Market Price Target')

# Dynamic Lines (to be updated)
line_vertical, = ax.plot([], [], 'r--', alpha=0.6, label='Projection')
line_tangent, = ax.plot([], [], 'g-', linewidth=2, label='Newton Step')
point_curr, = ax.plot([], [], 'ro', label='Current Estimate')

def init():
    ax.set_xlim(0, 100)
    ax.set_ylim(0, max(prices_for_plot))
    line_vertical.set_data([], [])
    line_tangent.set_data([], [])
    point_curr.set_data([], [])
    return line_vertical, line_tangent, point_curr

def update(frame):
    # Determine which step of the iteration we are in
    # x_vals contains pairs of coordinates. 
    if frame >= len(x_vals):
        return line_vertical, line_tangent, point_curr

    curr_x_pair = x_vals[frame]
    curr_y_pair = y_vals[frame]

    # Update visual elements
    line_tangent.set_data(curr_x_pair, curr_y_pair)
    
    # Draw a point at the "current" guess (the start of the line)
    point_curr.set_data([curr_x_pair[0]], [curr_y_pair[0]])
    
    return line_vertical, line_tangent, point_curr

# Create Animation
# We slow down the interval (1000ms) so you can see the steps clearly
anim = animation.FuncAnimation(fig, update, frames=len(x_vals),
                               init_func=init, interval=1000, 
                               repeat=True, blit=False)

ax.legend(loc='lower right')

# Pydroid3 requires this to display the window
plt.show()
