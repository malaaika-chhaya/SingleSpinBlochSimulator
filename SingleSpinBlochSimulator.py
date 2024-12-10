import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from matplotlib.gridspec import GridSpec
#%matplotlib qt

# Constants
gamma = 42.58e6  # Gyromagnetic ratio for hydrogen in Hz/T

# Create figure with two columns (sliders on the left, 3D plot on the right)
fig = plt.figure(figsize=(50, 50))
gs = GridSpec(1, 2, width_ratios=[1, 2])  # 1:2 ratio between slider and plot sections

# Add 3D plot to the right
ax = fig.add_subplot(gs[1], projection='3d')

# Create axes for sliders on the left
ax_B1 = plt.axes([0.05, 0.7, 0.35, 0.05], facecolor='lightgoldenrodyellow')
ax_delta_omega_rf = plt.axes([0.05, 0.55, 0.35, 0.05], facecolor='lightgoldenrodyellow')
ax_T1 = plt.axes([0.05, 0.4, 0.35, 0.05], facecolor='lightgoldenrodyellow')
ax_T2 = plt.axes([0.05, 0.25, 0.35, 0.05], facecolor='lightgoldenrodyellow')
ax_M0 = plt.axes([0.05, 0.1, 0.35, 0.05], facecolor='lightgoldenrodyellow')

# Initialize sliders
s_B1 = Slider(ax_B1, 'B1 (T)', 1e-6, 1e-4, valinit=1e-5)
s_delta_omega_rf = Slider(ax_delta_omega_rf, 'Δω_rf (rad/s)', 0, 500, valinit=100)
s_T1 = Slider(ax_T1, 'T1 (s)', 0.001, 0.1, valinit=0.01)
s_T2 = Slider(ax_T2, 'T2 (s)', 0.01, 1, valinit=0.1)
s_M0 = Slider(ax_M0, 'M0', 0, 1, valinit=1)

# Initial parameters from sliders
B1_init = s_B1.val
delta_omega_rf_init = s_delta_omega_rf.val
T1_init = s_T1.val
T2_init = s_T2.val
M0_init = s_M0.val

# Effective field in rotating frame
B_rot = np.array([B1_init, 0, delta_omega_rf_init / gamma])

# To capture 1/10th of the fastest dynamic
N_steps = 10

# Function to calculate the norm of the rotating frame field
def calc_time_step(B_rot):
    norm_B = np.linalg.norm(B_rot)
    if norm_B == 0:
        return np.inf  
    T = 1 / (gamma * norm_B)  # period
    return min(T, 1 / T1_init, 1 / T2_init)  # Time step based on the fastest dynamic

# Function to calculate nutation and relaxation
def rotation_matrix(B_rot, dt):
    """Compute the rotation matrix for nutation."""
    norm_B = np.linalg.norm(B_rot)
    if norm_B == 0:
        return np.eye(3)  # No nutation if B_rot is zero
    theta = gamma * norm_B * dt
    ux, uy, uz = B_rot / norm_B  # Unit vector
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return np.array([[cos_theta + ux**2 * (1 - cos_theta), ux * uy * (1 - cos_theta) - uz * sin_theta, ux * uz * (1 - cos_theta) + uy * sin_theta],
                     [uy * ux * (1 - cos_theta) + uz * sin_theta, cos_theta + uy**2 * (1 - cos_theta), uy * uz * (1 - cos_theta) - ux * sin_theta],
                     [uz * ux * (1 - cos_theta) - uy * sin_theta, uz * uy * (1 - cos_theta) + ux * sin_theta, cos_theta + uz**2 * (1 - cos_theta)]])

def apply_relaxation(M, dt, T1, T2, M0):
    """Apply relaxation to the magnetization."""
    Mx, My, Mz = M
    Mx *= np.exp(-dt / T2)
    My *= np.exp(-dt / T2)
    Mz = Mz * np.exp(-dt / T1) + M0 * (1 - np.exp(-dt / T1))
    return np.array([Mx, My, Mz])

def bloch_step(M, R, dt, T1, T2, M0):
    """Perform one step of nutation followed by relaxation."""
    M_new = np.dot(R, M)  # Nutation step
    M_new = apply_relaxation(M_new, dt, T1, T2, M0)  # Relaxation step
    return M_new

# Time evolution and storage
M = np.array([0, 0, M0_init])
M_storage = [M]

# Initialize the magnetization arrow and small sphere
arrow, = ax.plot([0, M_storage[0][0]], [0, M_storage[0][1]], [0, M_storage[0][2]], color='r', linewidth=3)
sphere, = ax.plot([0], [0], [0], 'bo', markersize=10)  # Small sphere at the origin

# Initialize trail
trail_x, trail_y, trail_z = [M_storage[0][0]], [M_storage[0][1]], [M_storage[0][2]]
trail, = ax.plot(trail_x, trail_y, trail_z, color='k', linewidth=0.5)  # Thin black line for the trail

# Set limits and labels
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel('Mx')
ax.set_ylabel('My')
ax.set_zlabel('Mz')
ax.set_title('Magnetization Dynamics')

def update_frame(i):
    """Update function for FuncAnimation."""
    arrow.set_data([0, M_storage[i][0]], [0, M_storage[i][1]])
    arrow.set_3d_properties([0, M_storage[i][2]])

    trail_x.append(M_storage[i][0])
    trail_y.append(M_storage[i][1])
    trail_z.append(M_storage[i][2])
    
    trail.set_data(trail_x, trail_y)
    trail.set_3d_properties(trail_z)

    return arrow, trail

def run_simulation():
    """Run the simulation and store magnetization dynamics."""
    global M_storage
    B1_init = s_B1.val
    delta_omega_rf_init = s_delta_omega_rf.val
    T1_init = s_T1.val
    T2_init = s_T2.val
    M0_init = s_M0.val
    
    # Update B_rot based on the slider values
    B_rot = np.array([B1_init, 0, delta_omega_rf_init / gamma])
    
    # Calculate time step and number of steps
    dt = calc_time_step(B_rot) / N_steps
    t_total = 1  # Total simulation time in seconds
    n_steps = int(t_total / dt)
    
    # Reset initial magnetization and storage
    M = np.array([0, 0, M0_init])
    M_storage = [M]
    
    for i in range(n_steps):
        M = bloch_step(M, rotation_matrix(B_rot, dt), dt, T1_init, T2_init, M0_init)
        M_storage.append(M)

    M_storage = np.array(M_storage)

# Update simulation when sliders are adjusted
def update_simulation(val):
    run_simulation()
    anim.event_source.start()  # Restart the animation after the sliders are adjusted

# Connect sliders to the update function
s_B1.on_changed(update_simulation)
s_delta_omega_rf.on_changed(update_simulation)
s_T1.on_changed(update_simulation)
s_T2.on_changed(update_simulation)
s_M0.on_changed(update_simulation)

# Run the initial simulation
run_simulation()

# Create animation
anim = FuncAnimation(fig, update_frame, frames=len(M_storage), interval=50, blit=True)

# Show plot
plt.show()