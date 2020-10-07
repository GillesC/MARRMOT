import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plotly.graph_objs as go
from plotting import LatexifyMatplotlib as lm

# Read data file and plot

df = pd.read_csv('antenna_pattern.txt', sep = "\s+|\t+|\s+\t+|\t+\s+")



df_2d = df.query('Phi==90')


ax = plt.subplot(111, projection='polar')
ax.scatter(df_2d["Theta"].values+180, df_2d['dBi'].values)
ax.grid(True)

ax.set_title("A line plot on a polar axis", va='bottom')
plt.show()

fig = go.Figure()
fig.add_trace(go.Scatterpolar(
        r = df_2d['dBi'].values,
        theta = df_2d["Theta"].values,
        mode = 'lines',
    ))
fig.write_html("Result/radiation_pattern_polar.html")


ax = plt.subplot(111, projection='polar')
ax.scatter(df_2d["Theta"], df_2d['dBi'].values)
ax.grid(True)

ax.set_title("H-plane")
plt.show()

df_e_plane = df.query('Phi==0')
df_h_plane = df.query('Phi==90')

ax = plt.subplot(111, projection='polar')

th = df_e_plane["Theta"]
y = df_e_plane['dBi'].values
y = y+ abs(y.min())
th = [x if x > 0 else abs(x) + 180 for x in th]

zipped = zip(th, y)
s = sorted(zipped, key = lambda t: t[0])
x,y = list(zip(*s))

ax.scatter(x,y)
ax.grid(True)

ax.set_title("E-plane")
plt.show()

ax = plt.subplot(111, projection='polar')
plt.polar(th, 10**(df_e_plane['dBi'].values/10))
plt.ylim(np.min(th),np.max(th))
ax.set_rscale('log')
plt.show()

plt.figure()
r_e = df_e_plane['dBi'].values
r_h = df_h_plane['dBi'].values

th_e = np.deg2rad(df_e_plane["Theta"])
th_h = np.deg2rad(df_h_plane["Theta"])

ax = plt.subplot(111, polar=True, projection='polar')
ax.set_theta_zero_location('N')
ax.set_theta_direction('clockwise')

plt.polar(th_e, r_e, label="E-plane")
for t,r in zip(th_e, r_e):
    print(f"{np.rad2deg(t)} {r}\\\\")
print()
plt.polar(th_h, r_h, label="H-plane")
ax.set_thetalim(0, 2*np.pi)
ax.legend()
lm.save("radiation-pattern.tex", scale_legend=False, show=True, plt=plt)



# plt.figure()
#
# th = np.deg2rad(df_h_plane["Theta"])
# ax = plt.subplot(111, polar=True, projection='polar')
# ax.set_theta_zero_location('N')
# ax.set_theta_direction('clockwise')
# plt.polar(th, r, )
# ax.set_thetalim(0, 2*np.pi)
# ax.set_rlim((r.min(), r.max()))
# plt.title("phi = 90")
# plt.legend()
# lm.save("phi-90-radiation-pattern.tex", scale_legend=0.7, show=True, plt=plt)

thetas = np.deg2rad(df['Theta'].values)
phis = np.deg2rad(df['Phi'].values)


x_shape = len(np.unique(thetas))
y_shape = len(np.unique(phis))

thetas = np.reshape(thetas, newshape=(x_shape,y_shape))
phis = np.reshape(phis, newshape=(x_shape,y_shape))

# unique_thetas = np.unique(thetas)
# unique_phis = np.unique(phis)
#
# THETA, PHI = np.meshgrid(unique_thetas, unique_phis)
# R =
power = np.reshape(10*(df['dBi'].values/10), newshape=(x_shape,y_shape))

R = power
X = R * np.sin(phis) * np.cos(thetas)
Y = R * np.sin(phis) * np.sin(thetas)
Z = R * np.cos(phis)

X_rad = X
Y_rad = Y
Z_rad = Z

fig = plt.figure()

ax = fig.add_subplot(1, 1, 1, projection='3d')


import matplotlib.colors as mcolors
cmap = plt.get_cmap('jet')
norm = mcolors.Normalize(vmin=Z.min(), vmax=Z.max())

plot = ax.plot_surface(
    X, Y, Z, rstride=1, cstride=1,
    facecolors=cmap(norm(Z)),
    linewidth=0, antialiased=False, alpha=0.5)


# Add Spherical Grid


phi, theta = np.linspace(0, 2 * np.pi, 40), np.linspace(0, np.pi, 40)

PHI, THETA = np.meshgrid(phi, theta)

R = np.max(power)

X = R * np.sin(THETA) * np.cos(PHI)

Y = R * np.sin(THETA) * np.sin(PHI)

Z = R * np.cos(THETA)

ax.plot_wireframe(X, Y, Z, linewidth=0.5, rstride=3, cstride=3)

plt.show()



surface = go.Surface(x=X_rad, y=Y_rad, z=Z_rad)
data = [surface]

fig = go.Figure(data=data)
fig.write_html("Result/radiation_pattern.html")


df = pd.read_csv('antenna_pattern.txt', sep = "\s+|\t+|\s+\t+|\t+\s+")

def cart2sph1(x,y,z):
    r = np.sqrt(x + y + z**2)
    phi = np.arctan2(y,x)
    th = np.arccos(z/r)
    return r, th, phi

def sph2cart1(r,th,phi):
    x = r * np.cos(phi) * np.sin(th)
    y = r * np.sin(phi) * np.sin(th)
    z = r * np.cos(th)
    return x, y, z

THETA = df["Theta"]
PHI = df["Phi"]
R = 10**(df["dBi"]/10)

phiSize = len(df["Phi"].unique())
thetaSize = len(df["Theta"].unique())

X = R * np.sin(PHI) * np.cos(THETA)
Y = R * np.sin(PHI) * np.sin(THETA)
Z = R * np.cos(PHI)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X = np.ones((phiSize, thetaSize))                                                                           # Prepare arrays to hold the cartesian coordinate data.
Y = np.ones((phiSize, thetaSize))
Z = np.ones((phiSize, thetaSize))
color_weight = np.ones((phiSize, thetaSize))
txt = [[0 for y in range(thetaSize)] for x in range(phiSize)]

min_dBi = np.abs(df["dBi"].min())

for phi_idx, phi in enumerate(np.unique(df["Phi"])):
    for theta_idx, theta in enumerate(np.unique(df["Theta"])):
        e = df.query(f"Phi=={phi} and Theta=={theta}").iloc[0]["dBi"]
        e_norm = min_dBi + e # so we dont have any negative numbers
        xe, ye, ze = sph2cart1(e_norm, math.radians(theta), math.radians(phi))                                   # Calculate cartesian coordinates

        X[phi_idx, theta_idx] = xe                                                                                  # Store cartesian coordinates
        Y[phi_idx, theta_idx] = ye
        Z[phi_idx, theta_idx] = ze
        color_weight[phi_idx, theta_idx] = e_norm
        txt[phi_idx][theta_idx] = f"phi: {phi} <br>theta: {theta}<br>{e:.2f} dBi"

ax.plot_surface(X, Y, Z, color='b')                                                                         # Plot surface
plt.ylabel('Y')
plt.xlabel('X')                                                                                             # Plot formatting
plt.show()

cmap = plt.get_cmap('viridis')
N = (color_weight-color_weight.min())/color_weight.max()

fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, surfacecolor=N, text = txt, hovertext=txt)])
fig.write_html("Result/radiation_pattern.html")

fig = plt.figure()
import matplotlib.colors as colors

ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
    facecolors=cmap(N),
    linewidth=0, antialiased=False, shade=False)
plt.show()
