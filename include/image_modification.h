#ifndef IMAGE_MODIFICATION_H
#define IMAGE_MODIFICATION_H

#include "image_common.h"

img* grayscaling(img *inputImg);        // szürkeárnyalatosítás
img* gaussianBlurring(img *inputImg);   // Gauss-elmosás (szürkeárnyalatos input)


#endif // IMAGE_MODIFICATION_H