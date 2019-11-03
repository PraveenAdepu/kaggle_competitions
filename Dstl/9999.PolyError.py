# -*- coding: utf-8 -*-
"""
Created on Mon Mar 06 19:24:49 2017

@author: SriPrav
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import shapely
from shapely.wkt import loads as wkt_loads

# Read the training data from WKT format
inDir = 'C:/Users/SriPrav/Documents/R/20Dstl'
df = pd.read_csv(inDir + '/submissions/Prav_sub133.csv',
        names=['ImageId', 'ClassType', 'MultipolygonWKT'], skiprows=1)
trainImageIds = np.sort(df.ImageId.unique())
classes = range(1,11)

# Function to load polygons
def get_polygons(ImageId):
    '''
    Parameters
    ----------
    sceneId : str
        sceneId like "6010_0_4"

    Returns
    -------
    polygonsList : dict
        Keys are CLASSES
        Values are shapely polygons
        None if sceneId is missing from df
    '''
    df_Image = df[df.ImageId == ImageId]
    if len(df_Image) == 0:
        return None
    polygonsList = {}
    for cType in classes:
        polygonsList[cType] = wkt_loads(df_Image[df_Image.ClassType == cType].MultipolygonWKT.values[0])
    return polygonsList
    
# Locate invalid polygons in the training data
for ImageId in trainImageIds:
    pl = get_polygons(ImageId)
    for cType in classes:
        if not pl[cType].is_valid:
            # One of the polygons in this MultiPolygon is invalid
            for i, poly in enumerate(pl[cType]):
                if not poly.is_valid:
                    print('Scene {} Class {} Polygon {} is invalid'.format(ImageId, cType, i))
                    fixed_poly = poly.buffer(0)  # Fix invalid polygon
                    print('Polygon fixed? :', fixed_poly.is_valid)     

#Self-intersection at or near point 0.0064258720785973179 -0.0088151272321301249
#Self-intersection at or near point 0.0064258720785973179 -0.0088151272321301249
#Scene 6070_0_3 Class 6 Polygon 5 is invalid
#('Polygon fixed? :', True)

#Prav_sub14.csv
#Self-intersection at or near point 0.0063600807990000004 -0.0067151100319882999
#Self-intersection at or near point 0.0063600807990000004 -0.0067151100319882999
#Scene 6020_1_4 Class 6 Polygon 7 is invalid
#('Polygon fixed? :', True)
#Self-intersection at or near point 0.0084570424398751013 -0.0090093818767315991
#Self-intersection at or near point 0.0084570424398751013 -0.0090093818767315991
#Scene 6030_3_3 Class 1 Polygon 2 is invalid
#('Polygon fixed? :', True)

#Prav_sub133.csv
#Self-intersection at or near point 0.0032720838498763996 -0.00042171503982150061
#Self-intersection at or near point 0.0032720838498763996 -0.00042171503982150061
#Scene 6030_2_4 Class 2 Polygon 32 is invalid
#('Polygon fixed? :', True)
#Self-intersection at or near point 0.0091139437627738452 -0.0082161992470148881
#Self-intersection at or near point 0.0091139437627738452 -0.0082161992470148881
#Scene 6080_4_3 Class 6 Polygon 6 is invalid
#('Polygon fixed? :', True)