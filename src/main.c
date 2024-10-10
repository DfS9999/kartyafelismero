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

    img *blurred = gaussianBlurring(grayOriginal);
    saveImage(blurred, "blurred");



    //img *result = templateMatchingOnGrayscale(grayOriginal, grayTemplate);
    //saveImage(result, "matching_result");

    stbi_image_free(original->values);
    free(original);
    //stbi_image_free(template->values);
    //free(template);
    free(grayOriginal->values);
    free(grayOriginal);
    //free(grayTemplate->values);
    //free(grayTemplate);
    free(blurred->values);
    free(blurred);

    
    return 0;
}
