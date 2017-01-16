# encoding: UTF-8
# Copyright 2016 Bart Marseille
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('TkAgg') #'macosx' is the default on mac
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Create new Figure with black background
fig = plt.figure(figsize=(16, 8), facecolor='black')

# Add a subplot with no frame
ax = plt.subplot(111, frameon=False)

# Generate random data
data = np.random.uniform(0, 1, (64,250)) #(64, 75))
X = np.linspace(-1, 1, data.shape[-1])
G = 1.5 * np.exp(-4 * X * X)


# Generate line plots
lines = []
for i in range(len(data)):
    # Small reduction of the X extents to get a cheap perspective effect
    xscale = 1 - i / 200.
    # Same for linewidth (thicker strokes on bottom)
    lw = 1.5 - i / 100.0
    line, = ax.plot(xscale * X, i + G * data[i], color="w", lw=lw)
    lines.append(line)

# Set y limit (or first line is cropped because of thickness)
ax.set_ylim(-1, 70)

# No ticks
ax.set_xticks([])
ax.set_yticks([])

# 2 part titles to get different font weights
ax.text(0.5, 1.0, "BURST", transform=ax.transAxes, ha="right", va="bottom", color="w", family="sans-serif", fontweight="bold", fontsize=16)
ax.text(0.5, 1.0, "SYNCHRONIZATION", transform=ax.transAxes, ha="left", va="bottom", color="w", family="sans-serif", fontweight="light", fontsize=16)

exite = [(-1,0), (1,0), (0,-1), (0,1)]
inhibit = [(-2,0), (2,0), (0,-2), (0,2)] #[(-1,-1), (-1,1), (1,-1), (1,1)]
def getNeighborhoodSum(data, f):
    global exite
    global inhibit
    newdata = np.zeros(shape=data.shape)
    for (x,y), value in np.ndenumerate(data):
        newdata[x,y] =  data.take(exite, mode='wrap').sum()/f - data.take(inhibit, mode='wrap').sum()/(2*f)
    return newdata

def doTimestep(data):
    a = 0.3
    b = 0.390634155 #0.3876
    c = 0.003
    return c / (a * np.square(b - data) + c)

def update(*args):
    global data


    # scale neighbourhood side input
    f = 150
    yscale = 2.0
    newdata = getNeighborhoodSum(data, f)
    newdata += data
    data = doTimestep(newdata)

    # Update data
    for i in range(len(data)):
        lines[i].set_ydata(i + yscale * data[i])

    # Return modified artists
    return lines

# Construct the animation, using the update function as the animation
# director.
anim = animation.FuncAnimation(fig, update, interval=20)
plt.show()
