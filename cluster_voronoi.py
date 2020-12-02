#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 14:06:12 2020

@author: julien
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.cluster.hierarchy import fclusterdata

nb_points = 100
dim = 2

properties = {
    'region_vert_count': { 'pos': 0, 'factor': 1},
    'region_area': { 'pos': 1, 'threshold': 4, 'factor': 1}
    }
nb_prop = len(properties)

nb_clusters = 10

colors = np.random.random((nb_clusters, 3))

#point_cluster = np.uint8(np.random.random((nb_points))
#                         * nb_clusters - 0.001)

nb_limit_points = 4
limit_points = [[100, 0],[-100, 0],[0, 100],[0, -100]]

points = np.random.random((nb_points + nb_limit_points, 
                           dim + nb_prop))
points2 = np.random.random((nb_points + nb_limit_points, 
                           dim + nb_prop))
points = (points + points2) / 2.0

for i in range(4):
    points[nb_points + i, :dim] = limit_points[i]

vor = Voronoi(points[:, :-nb_prop])

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x, np.roll(y,1))-np.dot(y,np.roll(x,1)))

facets = np.empty((nb_points + nb_limit_points))
areas = np.empty((nb_points + nb_limit_points))
for p in range(nb_points):
    facets[p] = len(vor.regions[vor.point_region[p]])
    verts = vor.vertices[vor.regions[vor.point_region[p]]]
    areas[p] = PolyArea(verts[:,0], verts[:,1])
    
points[:, dim + properties['region_vert_count']['pos']] = facets

def MyMetric(p1, p2):
    diff = p1[:-nb_prop] - p2[:-nb_prop]
    diffArea = (p1[dim + properties['region_area']['pos']]
                - p2[dim + properties['region_area']['pos']])
    if np.abs(diffArea) < properties['region_area']['threshold']:
        diffArea = 0;
    diff = diff + properties['region_area']['factor'] * diffArea
    return np.vdot(diff, diff) ** 0.5

point_cluster = fclusterdata(points[:-nb_limit_points],
                             nb_clusters,
                             criterion='maxclust',
                             metric=MyMetric)
point_cluster = point_cluster-1

for p in range(nb_points):
    reg = np.array(vor.regions[vor.point_region[p]])
    fltr = reg!=-1
    #if np.sum(fltr) == len(region):
    plt.fill(vor.vertices[reg[fltr].data, 0], 
             vor.vertices[reg[fltr].data, 1],
             color=colors[point_cluster[p]], 
             edgecolor="white",
             linewidth=1)
    
plt.scatter(points[:,0], points[:,1], zorder=2, s=10, c="white")
plt.xlabel('X')
plt.xlim(xmin=0.0, xmax=1.0)
plt.ylabel('Y')
plt.ylim(ymin=0.0, ymax=1.0)
plt.show()