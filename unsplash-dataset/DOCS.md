# Unsplash+ dataset documentation

The Unsplash+ Dataset is a cleaned and labeled(ongoing process) version of unsplash dataset:

## dataset.csv

The `dataset.csv` contains the entire dataset of 25K image, one per row. It contains all null and non null cleaned and free from noise EXIF features with photo orientation (1 horizontal, 0 vertical)

## train - validation

`train.csv` and `validation.csv` is a slice of the `dataset.csv` which contains samples only with non null EXIF (exif_iso, exif_focal_length, exif_aperture_values, exif_exposure_time) features along with experimental labeling.

| Field                       | Description |
|-----------------------------|-------------|
| photo_id                       | ID of the Unsplash photo |
| exif_camera_make               | Camera make (brand) extracted from the EXIF data |
| exif_camera_model              | Camera model extracted from the EXIF data |
| exif_iso                       | ISO setting of the camera, extracted from the EXIF data |
| exif_aperture_value            | Aperture setting of the camera, extracted from the EXIF data |
| exif_focal_length              | Focal length setting of the camera, extracted from the EXIF data |
| exif_exposure_time             | Exposure time setting of the camera, extracted from the EXIF data |
| orientation                    | 0: vertical, 1 horizontal |
| iso_noise_label                | 0: exif_iso < 800, 1: exif_iso => 800 | 
| bokeh_label                    | 0: exif_aperture_value > 3.5, 1: exif_aperture_value <=3.5 |
| high_speed_label               | 0: exif_exposure_time < 250, 1: exif_iso => 250 |
| close_up_label                 | 0: exif_focal_length < 35, 1: exif_focal_length => 35 |


## test

`test.csv` contains samples with all the null EXIF features(exif_iso, exif_aperture_value, exif_focal_length, exif_exposure_time)
