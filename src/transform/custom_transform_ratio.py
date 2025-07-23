import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from augraphy import AugraphyPipeline, InkBleed, Brightness, NoiseTexturize


def get_augraphy_transform():
    return AugraphyPipeline(
            ink_phase=[
              InkBleed(
                intensity_range=(0.5, 0.9),
                kernel_size=(5, 5),
                severity=(0.2, 0.4)
              ),
              NoiseTexturize(
                  sigma_range=(8, 11),
                  turbulence_range=(2, 4),
                  p=0.8
              )],
            paper_phase=[],
            post_phase=[]
        )

def get_transform_norm_tensor(image_size=(512, 512), image_normalization={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}):
    return A.Compose([
        A.LongestMaxSize(max_size=max(image_size)),
        A.PadIfNeeded(min_height=image_size[0], min_width=image_size[1], border_mode=cv2.BORDER_CONSTANT, fill=[255, 255, 255]),
        A.Normalize(mean=image_normalization["mean"], std=image_normalization["std"]),
        ToTensorV2()
    ])


def get_transform_rotation(p=0.8):
    return A.Compose([
        A.Rotate(limit=160, p=p, border_mode=cv2.BORDER_CONSTANT, fill=[255, 255, 255]),
        A.RandomRotate90(p=p-0.3),
    ])

def get_transform_brightness(brightness_limit=(0,0.25), p=0.8):
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=brightness_limit, p=p),
    ])

def get_transform_blur(blur_limit=(2,3), p=0.8):
    return A.Compose([
        A.MotionBlur(blur_limit=blur_limit, p=p),
    ])

def get_transform_gaussNoise(std_range=(0.1, 0.2), p=0.8):
    return A.Compose([
        A.GaussNoise(std_range=std_range, p=p),
    ])

def get_transform_shadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 3), shadow_dimension=6, shadow_intensity_range=(0.2, 0.5), p=0.8):
    return A.Compose([
        A.RandomShadow(
          shadow_roi=shadow_roi, 
          num_shadows_limit=num_shadows_limit, 
          shadow_dimension=shadow_dimension, 
          shadow_intensity_range=shadow_intensity_range, 
          p=p,
          ),
    ])

def get_horizontal_flip(p=1.0):
    return A.Compose([
        A.HorizontalFlip(p=p),
    ])

def get_vertical_flip(p=1.0):
    return A.Compose([
        A.VerticalFlip(p=p),
    ])

def get_transform_coarse_dropout(num_holes_range=[1, 2], hole_height_range=[0.1, 0.2], hole_width_range=[0.1, 0.12], fill=0, p=0.8):
    return A.Compose([
        A.CoarseDropout(
            num_holes_range=num_holes_range,
            hole_height_range=hole_height_range,
            hole_width_range=hole_width_range,
            fill=fill,
            p=p
        )
    ])

def get_grid_dropout(ratio=0.5,
                    random_offset=True,
                    shift_xy=[0, 0],
                    fill=0,
                    p=0.8):
    return A.Compose([
        A.GridDropout(ratio=ratio, random_offset=random_offset, shift_xy=shift_xy, fill=fill, p=p)
    ])

def get_transform_img_comp(compression_type='jpeg', quality_range=(20, 40), p=0.8):
    return A.Compose([
        A.ImageCompression(compression_type=compression_type, quality_range=quality_range, p=p),
    ])

