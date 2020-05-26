import ants
import os

dataDirectory = './InputData/'

fixedImage = ants.image_read('../mmwhs/CT_test/205_5/ct-image_crop_man_reg_resize/ct_train_1013_image.nii.gz')
fixedMask = ants.image_read( '../mmwhs/CT_test/205_5/ct-label_crop_man_reg_resize/ct_train_1013_label.nii.gz')
movingImage = ants.image_read( '../mmwhs/CT_test/205_5/ct-image_crop_man_reg_resize/ct_train_1014_image.nii.gz' )
movingMask = ants.image_read( '../mmwhs/CT_test/205_5/ct-label_crop_man_reg_resize/ct_train_1014_label.nii.gz' )

# Plot the fixed image/mask
ants.plot( fixedImage, overlay = fixedMask, overlay_cmap = "viridis", alpha = 0.9 )

# Plot the moving image/mask
ants.plot( movingImage, overlay = movingMask, overlay_cmap = "viridis", alpha = 0.9 )

#######
#
# Perform registration with no mask
#

outputDirectory = './OutputNoMaskANTsPy/'
if not os.path.isdir( outputDirectory ):
  os.mkdir( outputDirectory )

outputPrefix = outputDirectory + 'antsr'

registrationNoMask = ants.registration(
  fixed = fixedImage, moving = movingImage,
  type_of_transform = "SyNOnly",
  regIterations = ( 100, 75, 20, 0 ),
  verbose = True, outprefix = outputPrefix )

ants.image_write( registrationNoMask['warpedmovout'], outputPrefix + "Warped.nii.gz" )
ants.image_write( registrationNoMask['warpedfixout'], outputPrefix + "InverseWarped.nii.gz" )

# Plot the fixed and warped moving image
ants.plot( fixedImage, overlay = registrationNoMask['warpedmovout'], overlay_cmap = "viridis", alpha = 0.9 )

# Plot the moving and warped fixed image
ants.plot( movingImage, overlay = registrationNoMask['warpedfixout'], overlay_cmap = "viridis", alpha = 0.9 )

jacobian = ants.create_jacobian_determinant_image( fixedImage, registrationNoMask['fwdtransforms'][0] )
ants.plot( jacobian )

#######
#
# Perform registration with mask
#

outputDirectory = './OutputWithMaskANTsPy/'
if not os.path.isdir( outputDirectory ):
  os.mkdir( outputDirectory )

outputPrefix = outputDirectory + 'antsr'

registrationWithMask = ants.registration(
  fixed = fixedImage, moving = movingImage,
  mask = fixedMask,
  type_of_transform = "SyNOnly",
  regIterations = ( 100, 75, 20, 0 ),
  verbose = True, outprefix = outputPrefix )

ants.image_write(fixedImage,outputPrefix + "fix.nii.gz" )
ants.image_write( registrationWithMask['warpedmovout'], outputPrefix + "Warped.nii.gz" )
ants.image_write( registrationWithMask['warpedfixout'], outputPrefix + "InverseWarped.nii.gz" )

# Plot the fixed and warped moving image
ants.plot( fixedImage, overlay = registrationWithMask['warpedmovout'], overlay_cmap = "viridis", alpha = 0.9 )

# Plot the moving and warped fixed image
ants.plot( movingImage, overlay = registrationWithMask['warpedfixout'], overlay_cmap = "viridis", alpha = 0.9 )

jacobian = ants.create_jacobian_determinant_image( fixedImage, registrationWithMask['fwdtransforms'][0] )
ants.plot( jacobian )