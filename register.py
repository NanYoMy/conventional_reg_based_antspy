

import glob
import SimpleITK as sitk
import ants
#https://blog.csdn.net/weixin_43718675/article/details/102606717#2_antspy_92
#获取数据路径
import numpy as np
from evaluate import dice_compute,calculate_dice
import random
# def reg(fix_path,fix_label_path,move_path,move_label_path,type='SyN'):
# #读取数据，格式为： ants.core.ants_image.ANTsImage
#     fix_img = ants.image_read(fix_path)
#     fix_label_img= ants.image_read(fix_label_path)	
#     move_img = ants.image_read(move_path)
#     move_label_img = ants.image_read(move_label_path)	



#     g1 = ants.iMath_grad( fix_img )
#     g2 = ants.iMath_grad( move_img )


#     demonsMetric = ['demons', g1, g2, 1, 1]
#     ccMetric = ['CC', fix_img, move_img, 2, 4 ]
#     metrics = list( )
#     metrics.append( demonsMetric )

#     #配准
#     # outs = ants.registration(fix_img,move_img,type_of_transforme = 'Affine')
#     # outs = ants.registration( fix_img, move_img, type,  multivariate_extras = metrics )  
#     outs = ants.registration( fix_img, move_img, type)

#     #获取配准后的数据，并保存
#     reg_img = outs['warpedmovout']  
#     save_path = './warp_image.nii.gz'
#     ants.image_write(reg_img,save_path)

#     #获取move到fix的转换矩阵；将其应用到 move_label上；插值方式选取 最近邻插值; 这个时候也对应的将label变换到 配准后的move图像上
#     warp_label_img = ants.apply_transforms(fix_img ,move_label_img,transformlist= outs['fwdtransforms'],interpolator = 'nearestNeighbor')  
#     save_label_path = './warp_label.nii.gz'
#     ants.image_write(warp_label_img,save_label_path)
#     ants.image_write(fix_img,'./fix_img.nii.gz')
#     ants.image_write(move_img,'./mv_img.nii.gz')
#     ants.image_write(fix_label_img,'./fix_label.nii.gz')
#     dice_compute(warp_label_img.numpy().astype(np.int32),warp_label_img.numpy().astype(np.int32),indexes=[5])
#     ret=calculate_dice(np.where(warp_label_img.numpy()==5,1,0),np.where(fix_label_img.numpy()==5,1,0) )
#     print(ret)
#     return dice_compute(warp_label_img.numpy().astype(np.int32),fix_label_img.numpy().astype(np.int32),indexes=[5])
def reg(fix_path,fix_label_path,move_path,move_label_path,type='SyN'):
    #读取数据，格式为： ants.core.ants_image.ANTsImage
    fix_img = ants.image_read(fix_path)
    fix_label_img = ants.image_read(fix_label_path)
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
    # outs = ants.registration( fix_img, move_img, 'ElasticSyN',  multivariate_extras = metrics )  
    # outs = ants.registration( fix_img, move_img, type,verbose=True)
    outs = ants.registration( fix_img, move_img, type)

    #获取配准后的数据，并保存

    ants.image_write(outs['warpedmovout']  ,'./warp_image.nii.gz')
    if len(outs['fwdtransforms'])!=2:
        return [0]
    #获取move到fix的转换矩阵；将其应用到 move_label上；插值方式选取 最近邻插值; 这个时候也对应的将label变换到 配准后的move图像上
    reg_label_img = ants.apply_transforms(fix_img ,move_label_img,transformlist= outs['fwdtransforms'],interpolator = 'nearestNeighbor')  
    
    ants.image_write(reg_label_img,'./warp_label.nii.gz')
    ants.image_write(move_img,'./mv_img.nii.gz')
    ants.image_write(fix_img,'./fix_img.nii.gz')
    ants.image_write(fix_label_img,'./fix_label.nii.gz')
    return dice_compute(reg_label_img.numpy().astype(np.int32),fix_label_img.numpy().astype(np.int32),indexes=[5])

def test_registration(atlas_imgs,atlas_labs,target_imgs,target_labs,type):
    print(type)
    res=[]
    for target_img,target_lab in zip(target_imgs,target_labs):
        for atlas_img,atlas_lab in zip(atlas_imgs,atlas_labs):
            print("working:")
            print(atlas_img,atlas_lab,target_img,target_lab)
            ret=reg(target_img,target_lab,atlas_img,atlas_lab,type)
            print("result: "+str(ret[0]))
            res.append(ret[0])
    
    #
    no_zeros=[]
    for i in res:
        if i<0.01:
            continue
        else:
            no_zeros.append(i)

    res=np.array(no_zeros)
    print(res)
    print(np.mean(res))

# def test_registration_subdata(atlas_imgs,atlas_labs,target_imgs,target_labs,type):
#     print(type)
#     res=[]
#     for atlas_img,atlas_lab in zip(atlas_imgs[:5],atlas_labs[:5]):
#         for target_img,target_lab in zip(target_imgs[:5],target_labs[:5]):
#             print("working:")
#             print(atlas_img,atlas_lab,target_img,target_lab)
#             ret=reg(target_img,target_lab,atlas_img,atlas_lab,type)
#             print("result: "+str(ret[0]))
#             res.append(ret[0])
            
#     res=np.array(res)
#     print(res)
#     print(np.mean(res))

import SimpleITK

def de_parameter_nii(pathes,is_image=True):
    for path in pathes:
        img=sitk.ReadImage(path)

            
        array=sitk.GetArrayFromImage(img)
        if is_image:
        # img=sitk.Normalize(img)
            a_min,a_max=array.min(),array.max()
            array=(array-a_min)/(a_max-a_min)*250
        sitk.WriteImage(sitk.GetImageFromArray(array),path)


def init():
    atlas_ct_imgs=glob.glob('../mmwhs/CT_train/205_15/ct-image_crop_man_reg_resize/*.nii.gz')
    atlas_ct_labs=glob.glob('../mmwhs/CT_train/205_15/ct-label_crop_man_reg_resize/*.nii.gz')
    target_ct_imgs=glob.glob('../mmwhs/CT_test/205_5/ct-image_crop_man_reg_resize/*.nii.gz')
    target_ct_labs=glob.glob('../mmwhs/CT_test/205_5/ct-label_crop_man_reg_resize/*.nii.gz')

    atlas_mr_imgs=glob.glob('../mmwhs/MRI_train/205_15/mr-image_crop_man_reg_resize/*.nii.gz')
    atlas_mr_labs=glob.glob('../mmwhs/MRI_train/205_15/mr-label_crop_man_reg_resize/*.nii.gz')
    target_mr_imgs=glob.glob('../mmwhs/MRI_test/205_5/mr-image_crop_man_reg_resize/*.nii.gz')
    target_mr_labs=glob.glob('../mmwhs/MRI_test/205_5/mr-label_crop_man_reg_resize/*.nii.gz') 
    atlas_ct_imgs.sort()
    atlas_ct_labs.sort()
    target_ct_imgs.sort()
    target_ct_labs.sort()
    atlas_mr_imgs.sort()
    atlas_mr_labs.sort()
    target_mr_imgs.sort()
    target_mr_labs.sort()
    de_parameter_nii(atlas_ct_imgs)    
    de_parameter_nii(target_ct_imgs)    
    de_parameter_nii(atlas_ct_labs,False)    
    de_parameter_nii(target_ct_labs,False)    
    de_parameter_nii(atlas_mr_imgs)    
    de_parameter_nii(target_mr_imgs)    
    de_parameter_nii(atlas_mr_labs,False)    
    de_parameter_nii(target_mr_labs,False)    





if __name__=="__main__":
    # fix_path = 'mr_train_1001_image.nii.gz'   
    # fix_label_path = 'ct_train_1013_label.nii.gz'   
    # move_path = 'ct_train_1013_image.nii.gz'
    # move_label_path = 'ct_train_1013_label.nii.gz'
    # ret=reg(fix_path,fix_label_path,move_path,move_label_path,type='SyNCC')
    # print(ret)
    # init()
    atlas_ct_imgs=glob.glob('../mmwhs/CT_train/205_15/ct-image_crop_man_reg_resize/*.nii.gz')
    atlas_ct_labs=glob.glob('../mmwhs/CT_train/205_15/ct-label_crop_man_reg_resize/*.nii.gz')
    target_ct_imgs=glob.glob('../mmwhs/CT_test/205_5/ct-image_crop_man_reg_resize/*.nii.gz')
    target_ct_labs=glob.glob('../mmwhs/CT_test/205_5/ct-label_crop_man_reg_resize/*.nii.gz')

    atlas_mr_imgs=glob.glob('../mmwhs/MRI_train/205_15/mr-image_crop_man_reg_resize/*.nii.gz')
    atlas_mr_labs=glob.glob('../mmwhs/MRI_train/205_15/mr-label_crop_man_reg_resize/*.nii.gz')
    target_mr_imgs=glob.glob('../mmwhs/MRI_test/205_5/mr-image_crop_man_reg_resize/*.nii.gz')
    target_mr_labs=glob.glob('../mmwhs/MRI_test/205_5/mr-label_crop_man_reg_resize/*.nii.gz')
    

    atlas_ct_imgs.sort()
    atlas_ct_labs.sort()
    target_ct_imgs.sort()
    target_ct_labs.sort()
    atlas_mr_imgs.sort()
    atlas_mr_labs.sort()
    target_mr_imgs.sort()
    target_mr_labs.sort()

    
    # test_registration(atlas_ct_imgs,atlas_ct_labs,target_mr_imgs,target_mr_labs,type='SyN')
    # test_registration(atlas_mr_imgs,atlas_mr_labs,target_ct_imgs,target_ct_labs,type='SyN')

    # test_registration(atlas_ct_imgs,atlas_ct_labs,target_mr_imgs,target_mr_labs,type='ElasticSyN')
    # test_registration(atlas_mr_imgs,atlas_mr_labs,target_ct_imgs,target_ct_labs,type='ElasticSyN')

    # test_registration(atlas_ct_imgs,atlas_ct_labs,target_mr_imgs,target_mr_labs,type='ElasticSyN')
    # test_registration(atlas_mr_imgs,atlas_mr_labs,target_ct_imgs,target_ct_labs,type='ElasticSyN')

    # test_registration(atlas_ct_imgs,atlas_ct_labs,target_ct_imgs,target_ct_labs,type='ElasticSyN')
    # test_registration(atlas_mr_imgs,atlas_mr_labs,target_mr_imgs,target_mr_labs,type='ElasticSyN')
    test_registration(atlas_ct_imgs,atlas_ct_labs,target_mr_imgs,target_mr_labs,type='SyNOnly')
    test_registration(atlas_mr_imgs,atlas_mr_labs,target_ct_imgs,target_ct_labs,type='SyNOnly')
    # test_registration(atlas_ct_imgs,atlas_ct_labs,target_mr_imgs,target_mr_labs,type='ElasticSyN')
    # test_registration(atlas_mr_imgs,atlas_mr_labs,target_ct_imgs,target_ct_labs,type='ElasticSyN')
    # test_registration(atlas_mr_imgs,atlas_mr_labs,target_ct_imgs,target_ct_labs,type='SyN')
    # test_registration(atlas_mr_imgs,atlas_mr_labs,target_ct_imgs,target_ct_labs,type='DenseRigid')

    # test_registration(atlas_ct_imgs,atlas_ct_labs,target_mr_imgs,target_mr_labs,type='SyN')
    # test_registration(atlas_mr_imgs,atlas_mr_labs,target_ct_imgs,target_ct_labs,type='SyN')