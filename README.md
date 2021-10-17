This project used multiple image processing filters on this photo of my dog Bear. All of these are created and are not just uses of OpenCV functions. Open CV is just used for color conversion, image reading and 2d filtering/convolving. 


![grayBear](https://user-images.githubusercontent.com/77811085/137645026-d6e7e605-425b-453b-82e8-b532eafd2971.png)

The following photos are all outputs from the corresponding filters applied in the code.

Average Filter(Blur or image smoothing filter)

![averageBear](https://user-images.githubusercontent.com/77811085/137645051-05764e45-a6fa-445f-94f2-8beacf8c9928.png)


Horizontal Sobel Filter(1st Derivative Horizontal Edge Detection)

![sobelHorizBear](https://user-images.githubusercontent.com/77811085/137645099-ad048f56-67c7-49eb-836c-54b4db371a51.png)


Vertical Sobel Filter(1st Derivative Vertical Edge Detection)

![sobelVertBear](https://user-images.githubusercontent.com/77811085/137645116-df55b36d-867a-4b96-b96a-30b2d337c6d4.png)


Gradient Sobel Filter(Takes square root of the sum of vertical^2 and horizontal^2 filters)

![sobelGradientBear](https://user-images.githubusercontent.com/77811085/137645138-844e8c4b-1fe5-4e68-84a3-264b4601d382.png)


Laplacian Filter(2nd Derivative Edge Detection)

![LaplacianBear](https://user-images.githubusercontent.com/77811085/137645156-0df5bdf6-b5b9-4f70-a18a-beaa51e27b5e.png)

![LaplacianBearIn](https://user-images.githubusercontent.com/77811085/137645272-09450eb9-fe89-4635-b9a4-c450319c6128.png)


Median Filter(Noise Removal note: This one I modified to make a cool effect but not practical)

![medianBear](https://user-images.githubusercontent.com/77811085/137645308-273a6c54-44b4-423b-9151-a76d1e99da08.png)


Gaussian Filter(Blur or image smoothing filter)

![gaussianBlurBear](https://user-images.githubusercontent.com/77811085/137645318-1e8525fb-e7ea-4fa2-910f-53dfed32aeed.png)


Prewitt Vertical Filter(Another form of vertical edge detection)

![prewittVertBear](https://user-images.githubusercontent.com/77811085/137645477-c635bb96-6b68-4d46-a2f5-1059f8d28c55.png)


Prewitt Horizontal Filter(Another form of horizontal edge detection)

![prewittHorizBear](https://user-images.githubusercontent.com/77811085/137645489-631649fb-15c8-43f5-a0c7-80f218ca3ed8.png)


Prewitt Gradient Filter(Takes square root of the sum of vertical^2 and horizontal^2 filters)

![prewittGradientEdge](https://user-images.githubusercontent.com/77811085/137645542-9fca49ee-3d83-4a6f-9773-e392e0fa9d08.png)


Laplacian of Gaussian(Smoothed with Gaussian and detects fine edges with Laplacian note: this image was done with the originally sized image for better detail)

![lapgaussBear](https://user-images.githubusercontent.com/77811085/137645573-9daf7bd1-3017-42b6-9c0f-0aa0dc76243b.png)





