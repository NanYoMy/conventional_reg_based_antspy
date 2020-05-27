import ants


ori_img = ants.image_read('./ori3.nii.gz')
denoise=ants.denoise_image(ori_img)
ants.image_write(denoise,'./denoise3.nii.gz' )


