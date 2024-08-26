import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

# Create grids representing simplified cell structures
def create_cell_grid(size=10, p_empty=0.5):
    return np.random.choice([0, 1], size=(size, size), p=[p_empty, 1-p_empty])

# Set up the plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
high_lac_grid = create_cell_grid(p_empty=0.7)
low_lac_grid = create_cell_grid(p_empty=0.3)

im1 = ax1.imshow(high_lac_grid, cmap='binary')
im2 = ax2.imshow(low_lac_grid, cmap='binary')

ax1.set_title("High Lacunarity\n(More gaps)")
ax2.set_title("Low Lacunarity\n(Fewer gaps)")

# Add moving windows
window_size = 3
rect1 = Rectangle((0, 0), window_size, window_size, fill=False, ec='r', lw=2)
rect2 = Rectangle((0, 0), window_size, window_size, fill=False, ec='r', lw=2)
ax1.add_patch(rect1)
ax2.add_patch(rect2)

# Remove axis ticks
ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])

def update(frame):
    y, x = divmod(frame, high_lac_grid.shape[1] - window_size)
    rect1.set_xy((x, y))
    rect2.set_xy((x, y))
    return rect1, rect2

anim = FuncAnimation(fig, update, frames=range((high_lac_grid.shape[0] - window_size) * 
                                               (high_lac_grid.shape[1] - window_size)),
                     interval=200, blit=True)

plt.tight_layout()
plt.show()

# Uncomment the following line to save the animation
anim.save('lacunarity_concept.gif', writer='pillow', fps=5)