#include <stdio.h>
#include <stdlib.h>
#include "../include/image_IO.h"
#include "../include/image_modification.h"
#include "../include/image_recognition.h"


int main(int argc, char *argv[])
{
    if (argc < 2) {
        printf("Adj eleresi utat a kephez: %s <image.jpg> \n", argv[0]);
        return 1;
    }
    /*
    if (argc < 3) {
        printf("Adj eleresi utat a kephez: %s <image.jpg> <template.jpg> \n", argv[0]);
        return 1;
    }
    */
    img *original = loadImage(argv[1]);
    //img *template = loadImage(argv[2]);
    
    img *grayOriginal = grayscaling(original);
    saveImage(grayOriginal, "grayscale_original");
    //img *grayTemplate = grayscaling(template);

    int gray = 4;
    int count = gray * 3;
    size_t fSize = 32;
    char filename[fSize];
    img *images[count];
    img *startImg = grayOriginal;
    for (int i = 0; i < count; i += 3) {
        // gauss-filter
        images[i] = gaussianBlurring5x5(startImg);
        snprintf(filename, fSize, "gauss__%d", i/3);
        saveImage(images[i], filename);

        // sobel
        images[i + 1] = sobelEdgeDetection(images[i]);
        snprintf(filename, fSize, "sobel__%d", i/3);
        saveImage(images[i + 1], filename);

        // fél
        images[i + 2] = downsamplingBy2(images[i]);
        snprintf(filename, fSize, "felez__%d", i/3);
        saveImage(images[i + 2], filename);
        
        startImg = images[i + 2];
    }

    // free
    for (int i = 0; i < count; i += 3) {
        free(images[i]->values);
        free(images[i]);
    }
    
    return 0;
}
