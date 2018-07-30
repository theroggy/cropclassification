# -*- coding: utf-8 -*-
"""
Here are some general settings that are used by all cropclassification modules.

Passing all this by parameters became a bit messy.

@author: Pieter Roggemans
"""

# Column names used
id_column = 'CODE_OBJ'                       # Column name of the id column
class_column = 'classname'                   # Column name of the class
pixcount_s1s2_column = 'pixcount'            # Column name of the count of the number of pixels for sentinel1/2 images
prediction_column = 'pred1'                  # Column name of the standard prediction (probability can be same as other classes)
prediction_cons_column = 'pred_consolidated' # Column name of the the consolidated prediction (has a minimum probability)