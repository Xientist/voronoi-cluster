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
import open3d as o3d

# nombre de points à utiliser pour l'expérience
nb_points = 100
# nombre de dimension pour chaque points
dim = 2
# nombre de cluster maximum à générer
nb_clusters = 20
# incidence de la distance euclidienne sur la clusterization
euclid_factor = 1

#cloud = o3d.io.read_point_cloud("Hand.ply")

#npcloud = np.asarray(cloud.points)
#npcloud = npcloud[:nb_points] / 100 + 0.5
#npcloud[:,1] = npcloud[:,2]

# dictionnaire des propriété de voronoi qui nous intéressent
# elles posèdent une position (voir *1) et un facteur d'incidence
# sur la clusterization
properties = {
    'region_vert': { 'pos': 0, 'factor': .3},
    'region_area': { 'pos': 1, 'factor': 1}
    }
#le nombre de propriétés est calculé
nb_prop = len(properties)

# des couleurs aléatoires sont assignés aux k clusters
colors = np.random.random((nb_clusters, 3))

#point_cluster = np.uint8(np.random.random((nb_points))
#                         * nb_clusters - 0.001)

# détermination de points limites.
# la majorité des régions de Voronoi extérieures on une surface
# infinie. En définissant des points limites, autour de l'espace 
# de recherche, les régions extérieures de notre nuage deviennent
# intérieures et finies, bien que leurs surface soit toujours élevée
nb_limit_points = 8
limit_points = [[2, 0],[-2, 0],[0, 2],[0, -2],
                [2, 2],[-2, -2],[-2, 2],[2, -2]]

# la matrice des points est initialisée.
# Elle possède N points plus L points limites optionnels.
# chaque points est représentés sur D dimensions, plus P dimensions
# factices qui contiendront la valeur de chaque propriétés de la
# région de Voronoi du point
points = np.random.random((nb_points + nb_limit_points, 
                           dim + nb_prop))
#points2 = np.random.random((nb_points + nb_limit_points, 
#                           dim + nb_prop))
#points = (points + points2) / 2.0

#points[:-nb_limit_points,:dim] = npcloud[:,:dim]

# les points limites sont ajoutés
for i in range(nb_limit_points):
    points[nb_points + i, :dim] = limit_points[i]

# calcul du diagramme de Voronoi du nuage.
# La structure retournée par la fonction possède de nombreux éléments
# que nous utilisont par la suite
vor = Voronoi(points[:, :-nb_prop])

# fonction qui calcule la surface d'un polygone (2D)
def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x, np.roll(y,1))-np.dot(y,np.roll(x,1)))

# initialisation des structures des proptiétés
facets = np.empty((nb_points + nb_limit_points))
areas = np.empty((nb_points + nb_limit_points))

# calcul des propriétés
for p in range(nb_points):
    facets[p] = len(vor.regions[vor.point_region[p]])
    verts = vor.vertices[vor.regions[vor.point_region[p]]]
    areas[p] = PolyArea(verts[:,0], verts[:,1])

# les valeurs des propriétés pour chaque points sont stockés
# dans la matrice de points sur les dimensions P supplémentaires
points[:, dim + properties['region_vert']['pos']] = facets
points[:, dim + properties['region_area']['pos']] = areas

# (1*)
# la fonction de métrique utilisée pour la clusterization hiérachique.
# Impossible de savoir sur quel point l'on travaille, la fonction
# prends deux points quelconque et les compare pour obtenir une
# distance. Pour prendre en compte les propriétés de la région de
# Voronoi du point, ces propriétés calculées précédemment sont stocké
# dans les dimensions supplémentaires P définie dans l'algorithme.
# Un point sur 2 dimensions a donc 4 dimensions au total:
# Les dimensions (x,y) et, ici, les deux propriétés suivantes;
# le nombre de sommets de la région de Voronoi,
# la surface de la région de Voronoi
def MyMetric(p1, p2):
    
    # distance euclidienne
    diffDist = p1[:-nb_prop] - p2[:-nb_prop]
    diffDist = np.vdot(diffDist, diffDist) ** 0.5
    
    # différence du nombre de sommet des régions de p1 et p2
    diffVert = abs(p1[dim + properties['region_vert']['pos']]
                - p2[dim + properties['region_vert']['pos']])
    
    # différence des surfaces des région de p1 et p2
    diffArea = abs(p1[dim + properties['region_area']['pos']]
                - p2[dim + properties['region_area']['pos']])
    
    # combinaisons des différences
    diff = ( euclid_factor * diffDist
            + properties['region_vert']['factor'] * diffVert
            + properties['region_area']['factor'] * diffArea)
    
    return diff

# clusterization hiérarchique des points
point_cluster = fclusterdata(points[:-nb_limit_points],# L=0 -> [:]
                             nb_clusters,
                             criterion='maxclust',
                             metric=MyMetric)
# l'indexation est commencée à 1, on la décale à 0
point_cluster = point_cluster -1

# on calcule et affiche les nombres de points par clusters
nb_points_cluster = np.unique(point_cluster, return_counts=True)
print(nb_points_cluster[1])

# calcul et affichages de propriétés intéressantes des clusters;
# nombre de points de la région la plus grande,
# moyenne des points dans les régions,
# écart type du nombre de points par région,
# nombre de régions à un point
maxReg = max(nb_points_cluster[1])
# mean = np.mean(nb_points_cluster[1])# toujours la même pour N points 
std = np.std(nb_points_cluster[1])
ones = len(nb_points_cluster[1][nb_points_cluster[1]==1])

print('Points in biggest region: ' + str(maxReg))
#print('Average number of points in regions: ' + str(mean))
print('Std of number of points in regions: ' + str(std))
print('One-point regions: ' + str(ones))

# si le nombre de dimension D=2 on peut dessiner les régions colorées
# avec la couleur du cluster du point.
# On ignore les régions avec des segments infinis car on ne peut pas
# dessiner
for p in range(nb_points):
    reg = np.array(vor.regions[vor.point_region[p]])
    fltr = reg!=-1
    #if np.sum(fltr) == len(region):
    plt.fill(vor.vertices[reg[fltr].data, 0], 
             vor.vertices[reg[fltr].data, 1],
             color=colors[point_cluster[p]], 
             edgecolor="white",
             linewidth=1)

# les points du nuages sont aussi dessinés
plt.scatter(points[:,0], points[:,1], zorder=2, s=10, c="white")
plt.xlabel('X')
plt.xlim(xmin=-0.2, xmax=1.2)
plt.ylabel('Y')
plt.ylim(ymin=-0.2, ymax=1.2)
plt.show()