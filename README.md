# GlaucomaDetector
Detecting Glaucoma using Self Organizing Maps

This was my project under Dr.K.M.M Rao while in BITS-Pilani

## Steps to build
1. Place opencv version >=3.0 outside root directory.
2. Open project in Visual Studio 12 and start build.

## Running

image_path = input image path

image_mask_path = input image mask path

output_save_path = out image path

disk_mask_path = disk mask image generated for running Self Organizing Maps


### To train
    Glaucoma.exe T <image_path> <image_mask_path> <output_save_path> <disk_mask_path>

Trained data will be saved in som_data.txt

set the threshold for SOM in function [somThresh()](/glaucoma/Glaucoma.cpp#L152)
```
cv::Mat somThresh(cv::Mat &inp,cv::Mat &mask){
  ...
  ...
      if ((active>=<min_threshold>)&&(active<=max_threshold))
			{
  ...
  ...
```

### To run
    Glaucoma.exe O <image_path> <image_mask_path> <output_save_path> <disk_mask_path>

## Mask for fundus image
Create image of same size as input image and set Region of interest (ROI) pixels to '255' and rest to '0'.

Left fundus image, Right mask with ROI set to '255'
![Mask Image](/images/mask.jpg)

## Output
Program will out a single number which is cup-to-disk ratio. This ratio can be used to determine glaucoma. Ratio of 0.8 is a good indicator of glaucoma.

### Intermediate Images

Green channel -> Gaussian matched filter -> OTSU filter -> Morphological closing -> Sobel filter & Thresholding -> Hough circles -> Self-organising maps

![Intermediate Images](/images/intermediate.jpg)

### Final output image
Optic disk and cup marked.

![Output Image](/images/output.jpg)

## Author

* **Srijan Kumar Sharma** - [rajasrijan](https://github.com/rajasrijan)

## License

This project is licensed under the BSD 2-clause "Simplified" License - see the [LICENSE](LICENSE) file for details

## Paper
https://www.academia.edu/35315442/A_Project_Report_On_Biomedical_Imaging_for_Eye_Care
