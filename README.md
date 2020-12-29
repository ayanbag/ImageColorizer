# ImageColorizer
Transform black and white images into beautifully colored images using Deep Learning.

![result.jpg](images/result.png)

- Used pretrained model from [Zhang's Github](https://github.com/richzhang/colorization)  
- Original paper: [Colorful Image Colorization](https://arxiv.org/pdf/1603.08511.pdf)

The model used in this project is proposed by Zhang et al.’s 2016 ECCV paper, [Colorful Image Colorization](https://arxiv.org/pdf/1603.08511.pdf) where Zhang et al. decided to attack the problem of image colorization by using Convolutional Neural Networks to “hallucinate” what an input grayscale image would look like when colorized. All images are converted from the RGB color space to the Lab color space. Similar to the RGB color space, the Lab color space has three channels. But unlike the RGB color space, Lab encodes color information differently:

- The L channel encodes lightness intensity only
- The a channel encodes green-red.
- The b channel encodes blue-yellow

Since the L channel encodes only the intensity, we can use the L channel as our grayscale input to the network.

From there the network must learn to predict the a and b channels. Given the input L channel and the predicted ab channels we can then form our final output image.
  

# Requirement
- Python
- OpenCV 3.4.2+
- Numpy
  

# Usage

- Clone this Repository
```
git clone https://github.com/ayanbag/ImageColorizer.git
cd ImageColorizer
```

- **[Important]** Download the model from the following [link](https://drive.google.com/drive/folders/1hNvYYq9i7XYMhv9AtH9bFXRpxP8YcGo_?usp=sharing) and place it in the `model` folder

- Now excute the following command -
```
python imagecolorizer.py --image <path_to_image>
```
This script requires one arguments be passed to the script directly from the terminal, i.e. **--image** or **-i** which is the path to our input black/white image. All the colorized image will be saved in `output` folder