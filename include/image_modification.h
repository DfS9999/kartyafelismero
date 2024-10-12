#ifndef IMAGE_MODIFICATION_H
#define IMAGE_MODIFICATION_H

#include "image_common.h"

img* grayscaling(img *inputImg);                  // szürkeárnyalatosítás
img* gaussianBlurring3x3(img *inputImg);          // Gauss-elmosás (szürkeárnyalatos input) 3x3 kernel
img* gaussianBlurring5x5(img *inputImg);          // Gauss-elmosás (szürkeárnyalatos input) 5x5 kernel
img* downsamplingBy2(img *inputImg);              // kép-felezés (szürkeárnyalatos input)
img* sobelEdgeDetection(img *inputImg);           // Sobel él detektálás (szürkeárnyalatos input)

#endif // IMAGE_MODIFICATION_H