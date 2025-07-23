from augraphy import AugraphyPipeline, DelaunayTessellation, Folding, InkBleed, Brightness, NoiseTexturize, LinesDegradation, PatternGenerator

def ink_bleed(intensity_range=(0.7, 1.3), kernel_size=(5, 5), severity=(0.2, 0.4), p=1.0):
    return InkBleed(intensity_range=intensity_range, kernel_size=kernel_size, severity=severity)

def noise_texturize(sigma_range=(5, 7), turbulence_range=(2, 4), p=1.0):
    return NoiseTexturize(sigma_range=sigma_range, turbulence_range=turbulence_range, p=p)

def brightness(brightness_range=(0.9,1.25), p=1.0):
    return Brightness(brightness_range=brightness_range, p=p)

def lines_degradation(line_roi = (0.0, 0.0, 1.0, 1.0),
                      line_gradient_range=(16, 255),
                      line_gradient_direction= (2,2),
                      line_split_probability=(0.2, 0.7),
                      line_replacement_value=(250, 250),
                      line_min_length=(10, 10),
                      line_long_to_short_ratio = (2,10),
                      line_replacement_probability = (0.8, 0.8),
                      line_replacement_thickness = (1, 2), p=1.0):
    return LinesDegradation(line_roi=line_roi, line_gradient_range=line_gradient_range, line_gradient_direction=line_gradient_direction, line_split_probability=line_split_probability, line_replacement_value=line_replacement_value, line_min_length=line_min_length, line_long_to_short_ratio=line_long_to_short_ratio, line_replacement_probability=line_replacement_probability, line_replacement_thickness=line_replacement_thickness, p=p)

def folding(fold_count = 4, 
            fold_noise = 0.0, 
            fold_angle_range=(-10, 10), 
            gradient_width=(0.1, 0.2), 
            gradient_height=(0.01, 0.1),
            backdrop_color=(255, 255, 255), p=1.0):  
    return Folding(fold_count=fold_count, fold_noise=fold_noise, fold_angle_range=fold_angle_range, gradient_width=gradient_width, gradient_height=gradient_height, backdrop_color=backdrop_color, p=p)

def delaunay(n_points_range = (500, 800), n_horizontal_points_range=(50, 100), n_vertical_points_range=(50, 100), noise_type = "random", p=1.0):
    return DelaunayTessellation(n_points_range=n_points_range, n_horizontal_points_range=n_horizontal_points_range, n_vertical_points_range=n_vertical_points_range, noise_type=noise_type, p=p)

def pattern(imgx = 512, imgy= 512, n_rotation_range = (10,15), p=1.0):
  return PatternGenerator(imgx=imgx, imgy=imgy, n_rotation_range=n_rotation_range, p=p)

def get_augraphy_transform(p=1.0):
    return AugraphyPipeline(
        ink_phase=[ink_bleed(p=p), noise_texturize(p=p)],
        paper_phase=[],
        post_phase=[]
    )

