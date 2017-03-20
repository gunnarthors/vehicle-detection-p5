**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/colorhist.png
[image2]: ./output_images/bin_spat.png
[image3]: ./output_images/hog_example.png
[image4]: ./output_images/carnotcar.png
[image5]: ./output_images/trainingdata.png
[image6]: ./output_images/detection.png
[image7]: ./output_images/heatmap_example.png

---
All code is in CarND-Vehicle-Detection-P5.ipynb.
First code cell includes all the imports for the project.

###Histogram of Oriented Gradients (HOG)

The code for this step is contained in the second code cell of the IPython notebook CarND-Vehicle-Detection-P5.ipynb.
We use three features combined for the end single vector of the image. We extract the features from the image with spatial binning(bin_spatial function) combined with histogram of colors(color_hist function) and HOG(get_hog_features function). This is done through the function extract_features in same code cell.

The parameters we for extracting features are:
* colorspace = 'RGB2YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
* orient = 6
* pix_per_cell = 8
* cell_per_block = 2
* hog_channel = "ALL"
* spatial_size = (64, 64)
* hist_bins = 16

We tried multiple combination of parameters on the classifier(LinearSVM classifier explained later in the report) where we tried to get the maximum testing accuracy. So the parameters are mostly explored through the classifier and how he responded to some parameters changes.
Here is an example of histogram of colors using hist_bins=16

![alt text][image1]

Here is an example of spatial binning using size=(64,64)

![alt text][image2]

Here is an example of using HOG with orient=6, pix_per_cell=8, cell_per_block=2, hog_channel=ALL

![alt text][image3]

All those features are combined to a single vector.


We tried various combinations of parameters to maximize the accuracy of the classifier but in the end we accepted a bit lower accuracy on the test dataset as after trying the pipeline out on video in gave us fewer false positives and better results.


The code for next step is contained in the third code cell of the IPython notebook CarND-Vehicle-Detection-P5.ipynb.

We started with reading in all the images from the dataset. 
Here is an example of dataset images:

![alt text][image4]

Then we extract features for all the images to get a vector which we normalize using StandarScalar function from sklearn. 
We then split the dataset into training set and testing set which is split into 80%/20%.
We then train a linear support vector machine(LinearSVM) with a feature vector of length 15864. Training data is 14.208 examples and test set is 3.552 examples.

![alt text][image5]

Here we accepted an accuracy of 98.51% as it was giving better result on the video than a classifier with 99.5% accuracy. We believe that is because of the data and how it is set up for the classifier. So the accuracy could be a bit false. Many of the images in the dataset are similar so even though we random select images for the training and the testing set it can happen that alot of the images in both sets are similar. This could be solved with handpicking some images out of the dataset and select what should be in the set more carefully. It can take a lot of time which is something we don't have at this point so trial and errors to find the best fit has to be enough.


###Sliding Window Search

The code for this step is contained in the fourth, fifth and sixth code cell of the IPython notebook CarND-Vehicle-Detection-P5.ipynb.

We used Hog Sub-sampling Window Search method which allows us to only have to extract the Hog features once per frame. The find_car function extracts features and makes prediction. This function is in the fourth cell of the notebook. The find_cars only has to extract hog features once and then can be sub-sampled to get all of its overlaying windows. Each window is defined by a scaling factor where a scale of 1 would result in a window that's 8 x 8 cells then the overlap of each window is in terms of the cell distance.
For extracting features we used the same parameters as for the classifier which gave ok results. We make two searches per frame where we first search with scale 2.0 in the y axis from 390 to 646 with 2 cells per step which is 75% overlap and then we do a second search of scale 1.2 in the y axis region from 390 to 518 with 2 cells per step which is 75% overlap. We also ignored from 0 to 196 of the x axis in the image as we were getting a lot of detection on cars coming against us on the other lane.

Here is an example of detection on one frame:

![alt text][image6]

After making a prediction we create a heatmap in every frame. We keep a history of 11 heatmaps where we sum the history of heatmaps together to remove false positives and make sure we have a car to draw bounding box around. 

Here is an example of heatmap on same frame as the detection image:

![alt text][image7]


### Video Implementation

Here's a [link to my video result](https://youtu.be/aYDkidcjgz0)


###Discussion

This project is done with a lot of trials and errors in fine tuning the parameters both for the classifier and for the pipeline it self. The pipeline was showing a lot of false positives. The video is iterating 3 frames per second which is not acceptable but it is on CPU so it can be improved a lot with GPU. It can also be made a bit faster with less search boxes and more robust search. 
The parameter tuning is not perfect so the project can be improved a lot with better setup of the parameters. Also introducing a vehicle class which holds better history and calculates from history where the car will be next frame could be a great improvement. 


