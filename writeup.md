**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.jpg
[image2]: ./output_images/HOG_example.jpg
[image4]: ./output_images/sliding_window.jpg

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the code cell #1,2 of the IPython notebook located in "./main.ipynb".  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

The code for this step is contained in the code cell #3 of the IPython notebook located in "./main.ipynb".  

I tried to classify car images and non-car images with several HOG parameters in following ranges:

* orientations : [6, 9, 12]
* pixel per cell : [4, 8, 16]
* cell per block : [1, 2, 3]

As a result of comparison in accuracy, I decided to use these parameters.

* orientations : 9
* pixel per cell : 8
* cell per block : 2

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is contained in the code cell #4 of the IPython notebook located in "./main.ipynb".  

I trained a linear SVM using hog features with following paramters.

* input : 3 channel of YCrCb
* orientations : 9
* pixel per cell : 8
* cell per block : 2

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for this step is contained in the code cell #5 of the IPython notebook located in "./main.ipynb".  

I impremented a sliding window search in the following manner.

* use only 400px to 656px in y-direction because that is the region that cars could appear
* scale the image
* slide search windows over the scaled image by 16 pixels (75% overlapped)
* predict whether or not a car appears in the image in the window

The window size here is the same size of the image used to train the SVM. That is 64x64.

I decided to use 1.5 and 2.0 as scale ratios after trying each ratios from 1.0 to 2.0.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features, which provided a nice result.  Here are some example images:

![alt text][image4]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code for this step is contained in the code cell #6 of the IPython notebook located in "./main.ipynb".  

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

At the first time I created a video, there weren't enough detections to draw bounding boxes. I thought I should have used previous frames to make a heatmap more reliable.  Then I came up with the idea to take an average of last 5 frames. And I set the threshold to take the detection found at least 4 times in last 5 frames. Then I've got better result.

But I think I could improve the performance of this classifiers by using color histograms, binned color values, and taking more training data sets.
