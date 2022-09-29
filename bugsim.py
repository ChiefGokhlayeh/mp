#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Bug robot simulator for lecture "Motion Planning".

This module contains a simple robot simulator for the lecture 
"Motion Planning" at the Hochschule Esslingen.

Copyright 2019, Thao Dang

Permission is hereby granted, free of charge, to any person obtaining 
a copy of this software and associated documentation files (the "Software"), 
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS 
IN THE SOFTWARE.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation, rc
from IPython.display import HTML


def normalizeAngle(x):
    """Converts a an angle (in rad) to the range 0..2pi.

    Args:
        x: angle in rad.

    Returns:
        A float value (in rad) in the the range 0..2pi that corresponds to x. 
    """

    return np.remainder(x, 2*np.pi)


def get_line_intersection(ray, seg):
    """Computes the intersection of two line segments.

    Args:
        ray: the first line segment, ray = [p0_x, p0_y, p1_x, p1_y] 
             where (p0_x, p0_y) defines the first, and (p1_x, p1_y)
             the second point.

    Returns:
        None if no collision has been detected and 
        the collision point (coll_x, coll_y) otherwise. 
    """

    p0_x, p0_y, p1_x, p1_y = ray
    p2_x, p2_y, p3_x, p3_y = seg

    s1_x = p1_x - p0_x
    s1_y = p1_y - p0_y
    s2_x = p3_x - p2_x
    s2_y = p3_y - p2_y

    d = -s2_x * s1_y + s1_x * s2_y

    if d == 0:
        # line segments are parallel
        if math.fabs(-s1_y*(p2_x - p0_x) + s1_x*(p2_y - p0_y)) > 1e-8:
            # lines do not intersect
            return None
        else:
            # assume that ray originates from p0_x, p0_y
            n = math.sqrt(s1_x*s1_x + s1_y*s1_y)
            d2 = s1_x*(p2_x - p0_x) + s1_y*(p2_y - p0_y)
            d3 = s1_x*(p3_x - p0_x) + s1_y*(p3_y - p0_y)
            d2 = d2/n
            d3 = d3/n

            if (d2 <= 0) and (d3 > 0):
                return (p0_x, p0_y)
            elif (d3 <= 0) and (d2 > 0):
                return (p0_x, p0_y)
            elif (d2 < d3) and (d2 <= 1) and (d2 >= 0):
                return (p2_x, p2_y)
            elif (d3 < d2) and (d3 <= 1) and (d3 >= 0):
                return (p3_x, p3_y)

            return None

    s = (-s1_y * (p0_x - p2_x) + s1_x * (p0_y - p2_y)) / d
    t = (s2_x * (p0_y - p2_y) - s2_y * (p0_x - p2_x)) / d

    if (s >= 0) and (s <= 1) and (t >= 0) and (t <= 1):
        # Collision detected
        coll_x = p0_x + (t * s1_x)
        coll_y = p0_y + (t * s1_y)
        return (coll_x, coll_y)

    return None


def get_line_rect_intersection_and_dist(x, y, vx, vy, xmin, ymin, xmax, ymax):
    """Computes the intersection of a line segment and a rectangle.

    Args:
        x, y, vx, vy: the line segment, with start point (x, y) and 
            end point (x+vx, y+vy).
        xmin, ymin, xmax, ymax: the rectangle, with top left corner 
            (xmin, ymin) and lower right corner (xmax, ymax).
            Note that xmin <= xmax and ymin <= ymax is expected.

    Returns:
        The closest point and the distance to the closest point if 
        an intersection exists: (closestPt, minDist).
        If no intersection exists, (None, math.inf) is returned.
    """

    if (vx == 0) and (vy == 0):
        if (ymin <= y <= ymax) and (xmin <= x <= xmax):
            return ((x, y), 0)
        else:
            return (None, math.inf)

    if vy == 0:
        if ymin <= y <= ymax:
            t1 = (xmin - x)/vx
            t2 = (xmax - x)/vx
            if t1 >= t2:
                t1, t2 = t2, t1

            if (t1 > 1) or (t2 < 0):
                return (None, math.inf)
            elif t1 < 0:
                return ((x, y), 0)
            else:
                return ((x+t1*vx, y), t1*vx)
        else:
            return (None, math.inf)

    elif vx == 0:
        if xmin <= x <= xmax:
            t1 = (ymin - y)/vy
            t2 = (ymax - y)/vy
            if t1 >= t2:
                t1, t2 = t2, t1

            if (t1 > 1) or (t2 < 0):
                return (None, math.inf)
            elif t1 < 0:
                return ((x, y), 0)
            else:
                return ((x, y+t1*vy), t1*vy)
        else:
            return (None, math.inf)

    else:
        tmin, coll = np.inf, None

        for border in [ymin, ymax]:
            t = (border - y)/vy
            if 0 <= t <= 1:
                u = x + t*vx
                if xmin <= u <= xmax:
                    # line segments intersect
                    if t < tmin:
                        tmin, coll = t, (u, border)

        for border in [xmin, xmax]:
            t = (border - x)/vx
            if 0 <= t <= 1:
                v = y + t*vy
                if ymin <= v <= ymax:
                    # line segments intersect
                    if t < tmin:
                        tmin, coll = t, (border, v)

        if coll is None:
            return (None, math.inf)
        else:
            return (coll, math.sqrt((coll[0]-x)**2 + (coll[1]-y)**2))


class BugSim:
    """The main simulator class (see course material for more info).

    Attributes:
        x, y, theta: the 2d position and orientation (in rad) of the robot
        view_range: the maximum viewing distance of the sensor.
        sensor_resolution: the angular resolution (in rad) of
                the sensor.
        safety_distance: the robot should keep that distance to
                obstacles. 
        goal: the target position on the board
        history: all previously visited poses of the robot, 
                general form: [[x0, y0, theta0], [x1, y1, theta1], ... ]
    """

    def __init__(self, objects, goal, view_range=50,
                 sensor_resolution=20/180*np.pi, safety_distance=5):
        """Inits the bug simulator.

        Args:
            objects: the obstacles as a list of rectangles, e.g.
                    [[-5, -5, 205, 0],
                     [-5, 200, 205, 205]]
                where each line has the format 
                    [xmin, ymin, xmax, ymax]
            goal: the target position, e.g. (5,6)
            view_range: the maximum viewing distance of the sensor. 
                Everything above that is indicated as np.inf.
            sensor_resolution: the angular resolution (in rad) of
                the sensor.
            safety_distance: the robot should keep that distance to
                obstacles. 
        """
        self.objects = objects
        self.view_range = view_range
        self.sensor_resolution = sensor_resolution
        self.safety_distance = safety_distance
        self.visual_radius = max(self.safety_distance, 3)
        self.goal = goal
        self.spawn(0, 0, 0)

    def __scan(self):
        """Compute distances to objects as numpy array.

        If no objects is in range, return np.inf.
        """
        phis = np.arange(0, 2*np.pi, self.sensor_resolution)
        N = len(phis)
        self.sensor_readings = np.zeros((N, 2))

        for i in range(N):
            phi = phis[i]
            vx = self.view_range*np.cos(phi+self.theta)
            vy = self.view_range*np.sin(phi+self.theta)

            closestPt, minDist = None, np.inf
            for obj in self.objects:
                xmin, ymin, xmax, ymax = obj

                # If start point is in the box, we have a collision
                # NOTE: This assumes we have rectangular, axis aligned rectangles
                if (xmin <= self.x <= xmax) and (ymin <= self.y <= ymax):
                    closestPt, minDist = (self.x, self.y), 0
                    break

                closestPt_, minDist_ = get_line_rect_intersection_and_dist(
                    self.x, self.y, vx, vy, xmin, ymin, xmax, ymax)
                if minDist_ < minDist:
                    closestPt, minDist = closestPt_, minDist_

            if closestPt is not None:
                self.sensor_readings[i, :] = [phi, minDist]
            else:
                self.sensor_readings[i, :] = [phi, np.inf]

    def spawn(self, x, y, theta):
        """Spawn the robot at initial pose (x, y, theta).
        """
        self.x = x
        self.y = y
        self.theta = theta

        self.hasCollided = False
        self.history = None
        self.__scan()

    def drawBoard(self):
        """Draws the board with obstacles and target position.
        """
        for obj in self.objects:
            xmin, ymin, xmax, ymax = obj
            plt.fill([xmin, xmin, xmax, xmax, xmin],
                     [ymin, ymax, ymax, ymin, ymin], 'g')
        plt.plot(self.goal[0], self.goal[1], 'yd')
        plt.grid(True)
        plt.tight_layout()

    def getPose(self):
        """Gets the current position and orientation of the robot. 

        Returns: 
            A list of values (x, y, theta) where theta is in rad.
        """
        return (self.x, self.y, self.theta)

    def getScan(self):
        """Returns the sensor readings as numpy array.

        Returns:
            A numpy array of Nx2-shape where 
                * N is the number of measurements
                * [:, 0] returns the relative angle phi (in rad)
                * [:, 1] returns the distance to the closes obstacle
        """
        return self.sensor_readings

    def getDistAtPhi(self, sensor_readings, phi):
        """Gives the closest distance measurement to relative direction phi (rad).

        The sensor data is given as sensor_readings.
        """
        phi = normalizeAngle(phi)
        idx = np.int(0.5 + phi/self.sensor_resolution)
        idx = idx % sensor_readings.shape[0]
        return sensor_readings[idx, 1]

    def showScan(self, sensor_readings):
        """Visualizes the sensor readings.
        """
        for i in range(sensor_readings.shape[0]):
            phi, dist = sensor_readings[i, 0], sensor_readings[i, 1]
            xstart = self.x + self.visual_radius*np.cos(phi+self.theta)
            ystart = self.y + self.visual_radius*np.sin(phi+self.theta)
            dist = min(self.view_range, dist)
            xend = self.x + dist*np.cos(phi+self.theta)
            yend = self.y + dist*np.sin(phi+self.theta)
            plt.plot([xstart, xend], [ystart, yend], 'b')

    def checkCollision(self):
        """Returns True iff the robot has collided with an obstacle.
        """
        if np.min(self.sensor_readings[:, 1]) <= 0:
            return True
        else:
            return False

    def showRobot(self):
        """Plot the robot at it's current location.
        """
        circle1 = plt.Circle((self.x, self.y), self.visual_radius, color='r')
        plt.gca().add_artist(circle1)
        plt.plot(self.x, self.y, 'ro')
        plt.plot([self.x, self.x+2*self.visual_radius*np.cos(self.theta)],
                 [self.y, self.y+2*self.visual_radius*np.sin(self.theta)], 'r')

        if self.checkCollision():
            umin, umax, vmin, vmax = plt.axis()
            plt.text(0.5*(umin+umax), 0.5*(vmin+vmax), 'COLLISION!',
                     fontsize=32, color='m', horizontalalignment='center')

    def forward(self, dist):
        """Move robot about dist in forward direction.

        Please note that the robot not move a collision has been detected.
        """
        if self.hasCollided:
            print('Collision detected!')
            return

        self.x += np.cos(self.theta)*dist
        self.y += np.sin(self.theta)*dist
        if self.history is None:
            self.history = [[self.x, self.y, self.theta]]
        else:
            self.history.append([self.x, self.y, self.theta])
        self.__scan()

        if self.checkCollision():
            self.hasCollided = True
            print('Collision detected!')

    def turn(self, dtheta):
        """Turn robot about dtheta (rad) in mathematically positive direction
        """
        if self.hasCollided:
            print('Collision detected!')
            return

        self.theta += dtheta
        if self.history is None:
            self.history = [[self.x, self.y, self.theta]]
        else:
            self.history.append([self.x, self.y, self.theta])

        self.__scan()

    def getNumMoves(self):
        """Get number of motion steps the robot has done since it has been spawned.
        """
        if self.history is None:
            return 0
        else:
            return len(self.history)

    def animate(self, skipFrames=1):
        """Create an animation from the current simulator history.

        To use this function, call (and do not forget the last line!):

            simulator = BugSim(...set params...)
            ... do your thing...

            anim = simulator.animate()
            rc('animation', html='jshtml')
            anim

        Args:
            skipFrames: num of frames to skip (use this in longer video sequences 
                since creating movies with matplotlib is very slow)
        """

        print('Preparing animation...')
        fig, ax = plt.subplots()
        plt.close()

        my_goal_plot, = ax.plot(self.goal[0], self.goal[1], 'yd')

        for obj in self.objects:
            xmin, ymin, xmax, ymax = obj
            my_obj_plot, = ax.fill([xmin, xmin, xmax, xmax, xmin],
                                   [ymin, ymax, ymax, ymin, ymin], 'g')

        history = np.asarray(self.history)

        circle1 = plt.Circle((history[0, 0], history[0, 1]),
                             self.visual_radius, color='r', animated=True)
        my_circle_obj = ax.add_artist(circle1)
        my_hist_plot, = ax.plot([], [], 'b.')

        ax.grid(True)
        ax.axis('equal')

        def my_animate(i):
            idx = (skipFrames+1)*i
            my_circle_obj.center = (history[idx, 0], history[idx, 1])
            my_hist_plot.set_data(history[:idx, 0], history[:idx, 1])

            return [my_circle_obj, my_hist_plot, ]

        anim = animation.FuncAnimation(fig, my_animate, frames=len(self.history)//(skipFrames+1),
                                       interval=50, blit=True)
        print('...done!')
        return anim
