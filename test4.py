import ants
#获取数据路径
fix_path = 'ct_train_1001_image.nii.gz'   
move_path = 'ct_train_1013_image.nii.gz'
move_label_path = 'ct_train_1013_label.nii.gz'

#读取数据，格式为： ants.core.ants_image.ANTsImage
fix_img = ants.image_read(fix_path)
move_img = ants.image_read(move_path)
move_label_img = ants.image_read(move_label_path)	


g1 = ants.iMath_grad( fix_img )
g2 = ants.iMath_grad( move_img )


demonsMetric = ['demons', g1, g2, 1, 1]
ccMetric = ['CC', fix_img, move_img, 2, 4 ]
metrics = list( )
metrics.append( demonsMetric )

#配准
# outs = ants.registration(fix_img,move_img,type_of_transforme = 'Affine')
outs = ants.registration( fix_img, move_img, 'SyNOnly',  multivariate_extras = metrics )  

#获取配准后的数据，并保存
reg_img = outs['warpedmovout']  
save_path = './warp_image.nii.gz'
ants.image_write(reg_img,save_path)

#获取move到fix的转换矩阵；将其应用到 move_label上；插值方式选取 最近邻插值; 这个时候也对应的将label变换到 配准后的move图像上
reg_label_img = ants.apply_transforms(fix_img ,move_label_img,transformlist= outs['fwdtransforms'],interpolator = 'nearestNeighbor')  
save_label_path = './warp_label.nii.gz'
ants.image_write(reg_label_img,save_label_path)
