import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

if np.size(sys.argv) < 2:
  print("Please specify the type of downconversion")
  exit()

type = sys.argv[1]

class Cartesian:
  def __init__(self, x, y, z):
    self.x, self.y, self.z = x, y, z

  def __str__(self):
    return "{}, {}, {}".format(self.x, self.y, self.z)

  def __neg__(self):
    return Cartesian(-self.x, -self.y, -self.z)

  def __rmul__(self, a):
    return Cartesian(a*self.x, a*self.y, a*self.z)
  
  def __add__(self, c):
    return Cartesian(self.x+c.x, self.y+c.y, self.z+c.z)

  def __sub__(self, c):
    return self + -c


def drawLine (ax, c1, c2, color):
  x=[c1.x,c2.x]
  y=[c1.y,c2.y]
  z=[c1.z,c2.z]
  ax.plot(x,y,z,color=color)


def drawArrow (ax, origin, direction, color):
  ax.quiver(origin.x,origin.y,origin.z,direction.x,direction.y,direction.z,color=color)


def drawText (ax, c, text):
  ax.text(c.x,c.y,c.z,text,ha='center')


real_line = "#1f449c"
real_fill = "#7ca1cc"

imag_line = "#f05039"
imag_fill = "#e57a77"


def drawAxis (ax, origin, xlen, ylen, zlen):
  x0 = Cartesian(1,0,0)
  y0 = Cartesian(0,1,0)
  z0 = Cartesian(0,0,1)

  drawLine (ax, origin-xlen*x0, origin+xlen*x0, imag_line)
  drawLine (ax, origin-ylen*y0, origin+ylen*y0, 'k')
  drawLine (ax, origin-zlen*z0, origin+zlen*z0, real_line) 


def drawShape (ax, origin, ordaxis, absaxis, shape, ord1, ord2, fill_color, line_color):

  npts = np.size(shape)

  x = np.zeros(npts)
  y = np.zeros(npts)
  z = np.zeros(npts)
  
  steps = np.linspace(ord1, ord2, npts)

  for i in range(npts):
    pt = origin + steps[i] * ordaxis + shape[i] * absaxis
    x[i] = pt.x
    y[i] = pt.y
    z[i] = pt.z

  verts = [list(zip(x, y, z))]
  poly = Poly3DCollection(verts, alpha=0.6, color=fill_color)
  ax.add_collection3d(poly)
  ax.plot(x, y, z, color=line_color)


def Gaussian (x, centre, height, width):
  return height * np.exp(-((x - centre) / width) ** 2)

plt.rc('font', family='serif', serif='cm10', size=20)
plt.rc('text', usetex=True)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
ax.set_box_aspect([1,1,1])

limit = 1.5
ax.set_xlim3d(-limit, limit)
ax.set_zlim3d(-limit, limit)
ax.set_ylim3d(-limit, limit)

plt.axis('off')

elev = 30
azim = 35
roll = 0
ax.view_init(elev, azim, roll)

npts = 600
x = np.linspace(0, 1, npts)

hump1 = Gaussian (x, centre=.65, height=.30, width=.14)
hump2 = Gaussian (x, centre=.37, height=.12, width=.11)
Xw = hump1 + hump2

Sum_w = np.zeros(npts)
Dif_w = np.zeros(npts)
for iw in range(npts):
  Sum_w[iw] = Xw[iw] + Xw[npts-iw-1]
  Dif_w[iw] = Xw[iw] - Xw[npts-iw-1]

# ////////////////////////////////////////////////////////////////
#
# band-limited part of spectrum
#
# ////////////////////////////////////////////////////////////////

bpts = npts//3
bs = (npts - bpts)//2
be = (npts + bpts)//2

# fractional width of passband kept in band-limited signal
bl_npts = be-bs+1
bw = bl_npts / npts

# extract slices of spectra and place a zero on either end
bl_Xw = np.zeros(bl_npts+2)
bl_Xw[1:-2] = Xw[bs:be]

bl_Dif_w = np.zeros(bl_npts+2)
bl_Dif_w[1:-2] = Dif_w[bs:be]

bl_Sum_w = np.zeros(bl_npts+2)
bl_Sum_w[1:-2] = Sum_w[bs:be]

numax = 0.5 * (1+bw)
numin = 0.5 * (1-bw)

# delta function heights
delta = 0.4

lo = numin

if type == "lower":
  lo = numax
  delta = 0.5

# ////////////////////////////////////////////////////////////////
#
# set up axes in three dimensions
#
# ////////////////////////////////////////////////////////////////

# the three dimensions: frequency (nu), real (Re), and imaginary (Im)
nu = Cartesian(0,1,0)
Re = Cartesian(0,0,1)
Im = Cartesian(1,0,0)

# Get the projection matrix
proj = ax.get_proj()

# Extract 2D x and y directions from the projection matrix
ysep = 2.0 * Cartesian(proj[1, 0],proj[1, 1],proj[1, 2])
xsep = 1.5 * Cartesian(proj[0, 0],proj[0, 1],proj[0, 2])

# axis lengths
axlen = 0.25

# label heights
labh = .45
iqlabh = .6


if type == "nyquist1" or type == "nyquist2":

  Ox = 0.6 * ysep
  labh = 0.7
  delta = 0.5
  axis_length = 2.2

  if type == "nyquist1":
    label = r"$X_{u,1}(\nu)$"
    diglabel = r"$\hat{X}_{u,1}(\nu)$"
    nu1 = numin-lo
    nu2 = numax-lo
  else:
    label = r"$X_{u,2}(\nu)$"
    diglabel = r"$\hat{X}_{u,2}(\nu)$"
    nu1 = numin
    nu2 = numax

  # draw the band-limited down-converted signal
  drawShape (ax, Ox, nu, Re, bl_Xw, nu1, nu2, real_fill, real_line)
  drawShape (ax, Ox, nu, Im, bl_Xw, nu1, nu2, imag_fill, imag_line)

  drawShape (ax, Ox, nu, Re, bl_Xw, -nu1, -nu2, real_fill, real_line)
  drawShape (ax, Ox, nu, -Im, bl_Xw, -nu1, -nu2, imag_fill, imag_line)

  drawAxis (ax, Ox, axlen, axis_length, axlen)
  drawText (ax, Ox + labh*Re, label)

  # draw the sampling function
  for i in range(-2,3):
    shaw = Ox + i*2*bw*nu
    drawArrow (ax, shaw, delta*Re, real_line)
    drawArrow (ax, shaw, delta*Re, real_line)

  Ox = -Ox

  # draw the band-limited down-converted digitized signal
  for i in range(-2,3):
    shaw = Ox + i*2*bw*nu

    drawShape (ax, shaw, nu, Re, bl_Xw, nu1, nu2, real_fill, real_line)
    drawShape (ax, shaw, nu, Im, bl_Xw, nu1, nu2, imag_fill, imag_line)

    drawShape (ax, shaw, nu, Re, bl_Xw, -nu1, -nu2, real_fill, real_line)
    drawShape (ax, shaw, nu, -Im, bl_Xw, -nu1, -nu2, imag_fill, imag_line)

    # ... and the sampling function
    drawArrow (ax, shaw, delta*Re, real_line)
    drawArrow (ax, shaw, delta*Re, real_line)

  drawAxis (ax, Ox, axlen, axis_length, axlen)
  drawText (ax, Ox + labh*Re, diglabel)

  if np.size(sys.argv) > 2:
    plt.savefig(sys.argv[2], pad_inches=0)
  else:
    plt.show()
  exit()

# ////////////////////////////////////////////////////////////////
#
# original spectrum, X(w)
#
# ////////////////////////////////////////////////////////////////
Ox = 1.4 * ysep

drawShape (ax, Ox, nu, Im, Xw, 0, 1, imag_fill, imag_line)
drawShape (ax, Ox, nu, -Im, Xw, 0, -1, imag_fill, imag_line)
        
drawShape (ax, Ox, nu, Re, Xw, 0, 1, real_fill, real_line)
drawShape (ax, Ox, nu, Re, Xw, 0, -1, real_fill, real_line)

drawAxis (ax, Ox, axlen, 1.1, axlen)
drawText (ax, Ox + iqlabh*Re, r'$X(\nu)$')

if type == "dual":

  # local oscillator frequency on nu axis
  Cfpos = Ox + 0.5*nu
  Cfneg = Ox - 0.5*nu
  drawArrow (ax, Cfpos, delta*Re, real_line)
  drawArrow (ax, Cfneg, delta*Re, real_line)
  drawArrow (ax, Cfpos, -delta*Im, imag_line)
  drawArrow (ax, Cfneg, delta*Im, imag_line)

  # ////////////////////////////////////////////////////////////////
  #
  # in phase, I(w)
  #
  # ////////////////////////////////////////////////////////////////
  Oi = .4 * ysep - xsep

  drawShape (ax, Oi, nu, Re, Xw, .5, 1.5, real_fill, real_line)
  drawShape (ax, Oi, nu, Im, Xw, .5, 1.5, imag_fill, imag_line)

  drawShape (ax, Oi, nu, Re, Xw, -.5, -1.5, real_fill, real_line)
  drawShape (ax, Oi, nu, -Im, Xw, -.5, -1.5, imag_fill, imag_line)

  drawShape (ax, Oi, nu, Re, Sum_w, -.5, .5, real_fill, real_line)
  drawShape (ax, Oi, nu, Im, Dif_w, -.5, .5, imag_fill, imag_line)

  drawAxis (ax, Oi, axlen, 1.6, axlen)
  drawText (ax, Oi + iqlabh*Re, r"$I(\nu)$")

  # ////////////////////////////////////////////////////////////////
  #
  # quadrature, Q(w) 
  #
  # ////////////////////////////////////////////////////////////////
  Oq = .4 * ysep + xsep

  drawShape (ax, Oq, nu, -Re, Xw, .5, 1.5, imag_fill, imag_line)
  drawShape (ax, Oq, nu, Im, Xw, .5, 1.5, real_fill, real_line)

  drawShape (ax, Oq, nu, -Re, Xw, -.5, -1.5, imag_fill, imag_line)
  drawShape (ax, Oq, nu, -Im, Xw, -.5, -1.5, real_fill, real_line)

  drawShape (ax, Oq, nu, Re, Sum_w, -.5, .5, imag_fill, imag_line)
  drawShape (ax, Oq, nu, -Im, Dif_w, -.5, .5, real_fill, real_line)

  drawAxis (ax, Oq, axlen, 1.6, axlen)
  drawText (ax, Oq + iqlabh*Re, r"$Q(\nu)$")

  # ////////////////////////////////////////////////////////////////
  #
  # band-limited
  #
  # ////////////////////////////////////////////////////////////////

  # band-limited, in-phase spectrum
  Obi = -.6 * ysep - xsep
  drawShape (ax, Obi, nu, Im, bl_Dif_w, -.5*bw, .5*bw, imag_fill, imag_line)
  drawShape (ax, Obi, nu, Re, bl_Sum_w, -.5*bw, .5*bw, real_fill, real_line)
  drawAxis (ax, Obi, axlen, .5, axlen)
  drawText (ax, Obi + iqlabh*Re, r"$I_b(\nu)$")

  # band-limited, quadrature spectrum
  Obq = -.6 * ysep + xsep
  drawShape (ax, Obq, nu, Re, bl_Sum_w, -.5*bw, .5*bw, imag_fill, imag_line)
  drawShape (ax, Obq, nu, -Im, bl_Dif_w, -.5*bw, .5*bw, real_fill, real_line)
  drawAxis (ax, Obq, axlen, .5, axlen)
  drawText (ax, Obq + iqlabh*Re, r"$Q_b(\nu)$")

  # band-limited analytic signal
  Obx = -1.4 * ysep
  drawShape (ax, Obx, nu, Re, bl_Xw, -.5*bw, .5*bw, real_fill, real_line)
  drawShape (ax, Obx, nu, Im, bl_Xw, -.5*bw, .5*bw, imag_fill, imag_line)
  drawAxis (ax, Obx, axlen, .5, axlen)
  drawText (ax, Obx + labh*Re, r"$Z_b(\nu)$")

else:

  # ////////////////////////////////////////////////////////////////
  #
  # Band-limited signal, X_b
  #
  # ////////////////////////////////////////////////////////////////
  Ob = .5 * ysep

  drawShape (ax, Ob, nu, Re, bl_Xw, numin, numax, real_fill, real_line)
  drawShape (ax, Ob, nu, Im, bl_Xw, numin, numax, imag_fill, imag_line)

  drawShape (ax, Ob, nu, Re, bl_Xw, -numin, -numax, real_fill, real_line)
  drawShape (ax, Ob, nu, -Im, bl_Xw, -numin, -numax, imag_fill, imag_line)

  Cfpos = Ob + lo * nu
  Cfneg = Ob - lo * nu
  drawArrow (ax, Cfpos, delta*Re, real_line)
  drawArrow (ax, Cfneg, delta*Re, real_line)

  drawAxis (ax, Ob, axlen, 1.1, axlen)
  drawText (ax, Ob + iqlabh*Re, r"$X_b(\nu)$")
  
  Ou = -.5 * ysep

  # ////////////////////////////////////////////////////////////////
  #
  # Band-limited signal, X_b
  #
  # ////////////////////////////////////////////////////////////////
  drawShape (ax, Ou, nu, Re, bl_Xw, lo+numin, lo+numax, real_fill, real_line)
  drawShape (ax, Ou, nu, Im, bl_Xw, lo+numin, lo+numax, imag_fill, imag_line)

  drawShape (ax, Ou, nu, Re, bl_Xw, -(lo+numin), -(lo+numax), real_fill, real_line)
  drawShape (ax, Ou, nu, -Im, bl_Xw, -(lo+numin), -(lo+numax), imag_fill, imag_line)

  drawShape (ax, Ou, nu, Re, bl_Xw, numin-lo, numax-lo, real_fill, real_line)
  drawShape (ax, Ou, nu, Im, bl_Xw, numin-lo, numax-lo, imag_fill, imag_line)

  drawShape (ax, Ou, nu, Re, bl_Xw, lo-numin, lo-numax, real_fill, real_line)
  drawShape (ax, Ou, nu, -Im, bl_Xw, lo-numin, lo-numax, imag_fill, imag_line)

  drawAxis (ax, Ou, axlen, 1.6, axlen)
  drawText (ax, Ou + labh*Re, r"$X_m(\nu)$")
  
  # band-limited, baseband spectrum
  Obx = -1.5 * ysep

  # draw the band-limited baseband passband
  drawShape (ax, Obx, nu, Re, bl_Xw, numin-lo, numax-lo, real_fill, real_line)
  drawShape (ax, Obx, nu, Im, bl_Xw, numin-lo, numax-lo, imag_fill, imag_line)

  drawShape (ax, Obx, nu, Re, bl_Xw, lo-numin, lo-numax, real_fill, real_line)
  drawShape (ax, Obx, nu, -Im, bl_Xw, lo-numin, lo-numax, imag_fill, imag_line)

  drawAxis (ax, Obx, axlen, .5, axlen)

  if type == "upper":
    label = r"$X_u(\nu)$"
  else:
    label = r"$X_l(\nu)$"

  drawText (ax, Obx + labh*Re, label)

if np.size(sys.argv) > 2:
  plt.savefig(sys.argv[2], pad_inches=0)
else:
  plt.show()
