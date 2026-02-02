#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate a skymap with equatorial grid"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d

plt.rc('font', family='serif', serif='cm10', size=42)
plt.rc('text', usetex=True)

#plt.rcParams['text.usetex'] = True
#plt.rcParams['font.family'] = 'serif'
#plt.rcParams['font.serif'] = ['Computer Modern']
#plt.rcParams['font.size'] = 42

def draw_circle (ax, xc, yc, r, color='black'):
    phi = np.linspace(0, 2*np.pi, 200)
    x = xc + r * np.cos(phi)
    y = yc + r * np.sin(phi)
    ax.plot(x,y,color=color)

def draw_line (ax, x0, y0, x1, y1, color='black'):
    x = [x0, x1]
    y = [y0, y1]
    ax.plot(x,y,color=color)

def draw_grid (ax, xc, yc, r, nr, nt):

    for i in range(nr):
        radius = r*(i+1)/nr
        if i == nr//2+2:
          radius_arrow = radius
        draw_circle(ax,xc,yc,radius)

    for i in range(nt):
        theta = i*np.pi*2/nt
        if i == nt*5//8+1:
          theta_arrow = theta
        draw_line(ax,xc,yc,xc+r*np.cos(theta),yc+r*np.sin(theta))

    base_x = xc + radius_arrow * np.cos(theta_arrow)
    base_y = yc + radius_arrow * np.sin(theta_arrow)

    length = 0.3

    N_x = -length * np.cos(theta_arrow)
    N_y = -length * np.sin(theta_arrow)

    E_x = length * np.sin(theta_arrow)
    E_y = -length * np.cos(theta_arrow)

    hw = 0.1 # arrow head width
    hl = 0.15 # arrow head height
    ld_E = 1.7 # label displacement along arrow
    ld_N = 1.6

    ld_y = -0.05  # label y-displacement

    # North Arrow
    ax.arrow(base_x, base_y, N_x, N_y, color='black', head_width=hw, head_length=hl, linewidth=2)
    ax.text(base_x+ld_N*N_x, base_y+ld_N*N_y+ld_y, "N", ha='left', va='top', color='black')

    # East Arrow
    ax.arrow(base_x, base_y, E_x, E_y, color='black', head_width=hw, head_length=hl, linewidth=2)
    ax.text(base_x+ld_E*E_x, base_y+ld_E*E_y+ld_y, "E", ha='right', va='top', color='black')

# Draw some wavey lines
def draw_magnetic_field(ax, offset, height, theta):

    theta_rad = np.radians(theta)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)

    Npt = 100
    # Define the declination range for the field lines
    dec = np.linspace(-height/cos_theta, height/cos_theta, Npt)
    amplitude = 0.25
    spacing = 1.75
    N = 4
    frequency = 1.0
    phase = 2

    ra = amplitude * np.sin(2*np.pi*frequency/height * dec + phase)

    rap = cos_theta * ra + sin_theta * dec
    decp = -sin_theta * ra + cos_theta * dec

    hw = 0.3 # arrow head width
    hl = 0.5 # arrow head height

    iarrow = 11 

    # Plot the field lines
    for i in range(N):
        xrap = rap + offset + i*spacing
        ax.plot(xrap, decp, 'b-')

        idx = iarrow
        iarrow -= 8

        while idx < Npt:
            x = xrap[idx]
            y = decp[idx]
            dirx = xrap[idx+1] - x
            diry = decp[idx+1] - y
            ax.arrow(x, y, dirx, diry, fc='blue', ec='blue', head_width=hw, head_length=hl)
            idx += 30

    draw_electron(ax,offset+1.5,-1.5)
    draw_electron(ax,offset-1.8,-4.4)
    draw_electron(ax,offset+1,+1)
    draw_electron(ax,offset+1.5,+4.5)
    draw_electron(ax,offset+2.5,-2.5)
    draw_electron(ax,offset+3.0,+2.0)

def draw_electron(ax,x,y):
    el = 0.2
    ax.plot(x, y, 'bo')
    ax.text(x+el, y, '$e^-$',ha='left',va='center')

def draw_dish(ax,radius,height):

    curvature = height / radius**2
    vertex = 0.25/curvature

    print(f'{curvature=} {vertex=}')

    # Draw the x and y axes at the focus

    length = 5.0
    yfactor = 1.2
    ax.quiver(0,0,vertex, length,0,0, color='black')
    ax.text(length*yfactor,0,vertex,r'$\hat{\boldmath{y}}$',ha='center')

    xfactor = 1.3
    ax.quiver(0,0,vertex, 0,length,0, color='black')
    ax.text(0,length*xfactor,vertex,r'$\hat{\boldmath{x}}$',ha='center')

    # draw the parabolic reflector

    u = np.linspace(0, 2 * np.pi, 100)
    r = np.linspace(0, radius, 100)
    px = np.outer(np.cos(u), r)
    py = np.outer(np.sin(u), r)
    pz = np.outer(np.ones(np.size(u)), curvature*r**2)

    ax.plot_surface(px, py, pz, color='white')
    ax.plot_wireframe(px, py, pz, rstride=10, cstride=20, color='black')

    # Draw the conical slice focus cabin

    base_radius = 1.0
    top_radius = 0.75
    height = 1.0

    cx = np.outer(np.cos(u), (base_radius, top_radius))
    cy = np.outer(np.sin(u), (base_radius, top_radius))
    cz = np.outer(np.ones(np.size(u)), (vertex, vertex+height))

    ax.plot_surface(cx, cy, cz, color='white', alpha=1.0)

    # Draw a circle on the x=0 'wall'
    p = Circle((0, 0), top_radius, fc='white', ec='black')
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=vertex+height)

# Set the limits of the Cartesian axes *before* plotting anything
cart_x_min = -15  # Set your desired minimum x-value
cart_x_max = 9   # Set your desired maximum x-value
cart_y_min = -5  # Set your desired minimum y-value
cart_y_max = 5   # Set your desired maximum y-value

fig = plt.figure(figsize=[cart_x_max-cart_x_min, cart_y_max-cart_y_min])

ax = fig.subplots()

# plt.rc('font', family='serif', serif='cm10', size=28)

# Hide the ticks and labels of the Cartesian grid
ax.set_frame_on(False)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.set_aspect('equal')

ax.set_xlim(cart_x_min, cart_x_max)  # FIX: Set x-axis limits FIRST
ax.set_ylim(cart_y_min, cart_y_max)  # FIX: Set y-axis limits FIRST

offset = 4.5
height = 6
theta = 30 
draw_magnetic_field(ax, offset, height, theta)

ax3d = ax.inset_axes([-12, -7, 12, 12], transform=ax.transData, projection='3d')

ax3d.set_axis_off()
ax3d.set_box_aspect([1,1,1])

dish_depth=3
dish_radius=10

limit = dish_radius/np.sqrt(2)
ax3d.set_xlim3d(-limit, limit)
ax3d.set_ylim3d(-limit, limit)
ax3d.set_zlim3d(-limit, limit)

elev = 20
azim = 40
roll = -65

ax3d.view_init(elev, azim, roll)

draw_dish(ax3d, dish_radius, dish_depth)

ax2 = ax.inset_axes([-1, -4, 8, 8], transform=ax.transData)

limit = 0.8
xoffset = -0.1
yoffset = -0.4

# Set plot limits
ax2.set_xlim(xoffset-limit, xoffset+limit)
ax2.set_ylim(yoffset-limit, yoffset+limit)
ax2.set_aspect('equal')

ax2.set_frame_on(False)
ax2.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)

draw_grid(ax2, 3,3,6,10,51)

axi = ax.inset_axes([cart_x_min, -4, 8, 8], transform=ax.transData)

axi.set_frame_on(False)
axi.get_xaxis().set_visible(False)
axi.get_yaxis().set_visible(False)

circuit_img = mpimg.imread('circuit.png')
axi.imshow(circuit_img)

plt.rcParams['font.size'] = 56

height=5.1
ax.text(-13,height,"$\mathbf\Phi$")
ax.text(-9.5,height,"\\bf G")
ax.text(-6.0,height,"\\bf D")
ax.text(-2.5,height,"\\bf C")
ax.text(1.0,height,"\\bf P")
ax.text(4.5,height,"\\bf F")

plt.savefig("signal_path-raw.pdf")

