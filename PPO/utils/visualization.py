import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from reward import Axis
from math import asin, acos, sin, cos, exp, pi, tan
from random import random

fig = plt.figure()
ax = plt.axes(projection='3d')
line1,  = ax.plot([],[],[], color = 'red')
line2,  = ax.plot([],[],[], color = 'red')
line3,  = ax.plot([],[],[], color = 'black')
line4,  = ax.plot([],[],[], color = 'black')

class Visualization():
    def __init__(self):
        theta = np.linspace(0, 3.14*4, 200)
        # r = np.linspace(0, 1, 200)
        r = 0.5
        self.x = r * np.cos(theta)
        self.y = r * np.sin(theta)
        self.z = np.linspace(0, 2, 200)
        self.a = [0]
        self.b = [0]
        self.c = [0]
        self.trajectory = [np.array([[0  , 0   , -0.5, 1],
                                     [0.5, -0.5, 0   , 0],
                                     [0  , 0   , 0   , 0]])]
        for i in range(1, 200):
            d = ((self.x[i] - self.x[i-1])**2+(self.y[i] - self.y[i-1])**2)**0.5
            theta = asin((self.y[i] - self.y[i-1])/d)
            if acos((self.x[i] - self.x[i-1])/d) > pi/2:
                theta = pi - theta
            theta -= pi/2
            self.a.append(theta)
            d = ((self.z[i] - self.z[i-1])**2+(self.y[i] - self.y[i-1])**2+(self.x[i] - self.x[i-1])**2)**0.5
            self.b.append(asin((self.z[i] - self.z[i-1])/d))
            self.c.append(self.c[-1]+0.1*random()-0.2)
            axis = Axis(self.a[-1], self.b[-1], self.c[-1])
            self.trajectory.append(axis.transform(np.array([[0  , 0   , -0.03, 0.06],
                                                            [0.03, -0.03, 0   , 0],
                                                            [0  , 0   , 0   , 0]]), reverse=True))
        self.trajectory = np.array(self.trajectory)
        # print(np.array(self.a[1:])-np.array(self.a[:-1]))
        # print(self.b)
        # for i in range(200):
        #     speed = np.array([cos(self.a[i]), sin(self.a[i]), tan(self.b[i])])
        #     # speed = self.trajectory[i,:,1] - self.trajectory[i,:,0]
        #     test = self.trajectory[i,:,3] - self.trajectory[i,:,2]
        #     speed = speed/(speed@speed)**0.5
        #     test = test/(test@test)**0.5
        #     print(test, speed, test@speed)
    
    def update(self, t):
        ax.set_xlim(-0.5,0.5)
        ax.set_ylim(-0.5,0.5)
        ax.set_zlim(0,2)
        line1.set_data_3d(self.trajectory[1:t,0,0]+self.x[1:t], self.trajectory[1:t,1,0]+self.y[1:t], self.trajectory[1:t,2,0]+self.z[1:t])
        line2.set_data_3d(self.trajectory[1:t,0,1]+self.x[1:t], self.trajectory[1:t,1,1]+self.y[1:t], self.trajectory[1:t,2,1]+self.z[1:t])
        line3.set_data_3d([self.trajectory[t,0,0]+self.x[t], self.trajectory[t,0,1]+self.x[t]], [self.trajectory[t,1,0]+self.y[t],self.trajectory[t,1,1]+self.y[t]], [self.trajectory[t,2,0]+self.z[t], self.trajectory[t,2,1]+self.z[t]])
        line4.set_data_3d([self.trajectory[t,0,2]+self.x[t], self.trajectory[t,0,3]+self.x[t]], [self.trajectory[t,1,2]+self.y[t],self.trajectory[t,1,3]+self.y[t]], [self.trajectory[t,2,2]+self.z[t], self.trajectory[t,2,3]+self.z[t]])
        return line1, line2, line3, line4

# def getData(t):
#     theta = np.linspace(0, 3.14*4, 200)[:t]
#     r = np.linspace(0, 1, 200)[:t]
#     x = r * np.cos(theta)[:t]
#     y = r * np.sin(theta)[:t]
#     z = np.linspace(0, 2, 200)[:t]
#     return x, y, z

# def update(t):
#     ax.set_xlim(-0.5,0.5)
#     ax.set_ylim(-1,1)
#     x,y,z = getData(t)
#     line.set_data_3d(x,y,z)
#     return line

visual = Visualization()
ani = FuncAnimation(fig, visual.update, frames = np.arange(199), interval = 50)
ani.save("move.gif", writer = 'Pillow', fps = 10)
plt.show()