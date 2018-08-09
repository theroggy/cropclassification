# -*- coding: utf-8 -*-
"""
Here are some general settings that are used by all cropclassification modules.

Passing all this by parameters became a bit messy.

@author: Pieter Roggemans
"""

# Dedicated column names used in the parcel data files
id_column = 'CODE_OBJ'                       # Column name of the id column
class_column = 'classname'                   # Column name of the class, after preprocessing to optimize the classification
class_orig_column = 'classname_orig'         # Column name of the original class of the parcel, before additional preprocessing
is_eligible_column = 'is_eligible'           # Column that is 1 for parcels with an eligible crop, 0 for ineligible crop/landcover
is_permanent_column = 'is_permanent'         # Column that is 1 for parcels with a permanent landcover, 0 for a regular crop. Permanent landcovers can/should be followed up in the LPIS upkeep
pixcount_s1s2_column = 'pixcount'            # Column name of the count of the number of pixels for sentinel1/2 images

# List of all dedicated column names
dedicated_data_columns = [id_column, class_column, class_orig_column, is_eligible_column, is_permanent_column, pixcount_s1s2_column]

prediction_column = 'pred1'                  # Column name of the standard prediction (probability can be same as other classes)
prediction_cons_column = 'pred_consolidated' # Column name of the the consolidated prediction (has a minimum probability)
prediction_status = 'pred_status'            # The status/result of the prediction
