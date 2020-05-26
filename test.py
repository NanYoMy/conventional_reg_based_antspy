# -*- coding:utf-8 -*-
import ants
img1 = ants.image_read('../mmwhs/CT_test/205_5/ct-image_crop_man_reg_resize/ct_train_1013_image.nii.gz')
img2 = ants.image_read('../mmwhs/CT_test/205_5/ct-image_crop_man_reg_resize/ct_train_1014_image.nii.gz')
print(img2)
print(img1)
g1 = ants.iMath_grad( img1 )
g2 = ants.iMath_grad( img2 )

reg1 = ants.registration( img1, img2, 'SyNOnly' )
demonsMetric = ['demons', g1, g2, 1, 1]
ccMetric = ['CC', img1, img2, 2, 4 ]
metrics = list( )
metrics.append( demonsMetric )
print("start to registration...........")
reg2 = ants.registration( img1, img2, 'SyNOnly',  multivariate_extras = metrics )
metrics.append( ccMetric )
reg3 = ants.registration( img1, img2, 'SyNOnly',
    multivariate_extras = metrics )



print( ants.image_mutual_information( img1, img2 ) )
print( ants.image_mutual_information( img1, reg1['warpedmovout'] ) )
print( ants.image_mutual_information( img1, reg2['warpedmovout'] ) )
print( ants.image_mutual_information( img1, reg3['warpedmovout'] ) )



