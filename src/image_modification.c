#include "../include/image_modification.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h> // memcpy
//#include <omp.h> // OpenMP párhuzamosításhoz


img* grayscaling(img *inputImg)
{
    if (!inputImg || !inputImg->values) {
        printf("Ervenytelen bemeneti kep.\n");
        return NULL;
    }

    img* grayImg = (img*) malloc(sizeof(img));
    if (grayImg == NULL) {
        printf("Sikertelen memoriafoglalas szurkearnyalatoshoz.\n");
        return NULL;
    }

    grayImg->width = inputImg->width;
    grayImg->height = inputImg->height;
    grayImg->channels = 1;

    const int pixels = inputImg->height * inputImg->width;
    grayImg->values = (intensity*) malloc(pixels * sizeof(intensity));
    if (!grayImg->values) {
        printf("Sikertelen memoriafoglalas szurkearnyalatoshoz.\n");
        free(grayImg);
        return NULL;
    }

    if (inputImg->channels == 1) {
            memcpy(grayImg->values, inputImg->values, pixels * sizeof(intensity));
    }
    else if (inputImg->channels == 3 || inputImg->channels == 4){
        //const float coefficient_R = 0.3f;
        //const float coefficient_G = 0.59f;
        //const float coefficient_B = 0.11f;
        const int coefficient_R = (int) (0.3f  * 256);
        const int coefficient_G = (int) (0.59f * 256);
        const int coefficient_B = (int) (0.11f * 256);

        for (int i = 0; i < pixels; ++i) {
            // R0 G0 B0 R1 G1 B1
            const int idx = i * inputImg->channels;
            intensity r = inputImg->values[idx + 0];
            intensity g = inputImg->values[idx + 1];
            intensity b = inputImg->values[idx + 2];
            //grayImg->values[i] = r*coefficient_R + g*coefficient_G + b*coefficient_B;
            
            // osztás 2^8=256 -al
            grayImg->values[i] = (r*coefficient_R + g*coefficient_G + b*coefficient_B) >> 8;
        }
    }
    else {
        printf("Ervenytelen bemeneti kep (1, 3, 4 csatorna).\n");
        free(grayImg->values);
        free(grayImg);
        return NULL;
    }

    printf("Height: %d Width: %d Channels: %d. ", grayImg->height, grayImg->width, grayImg->channels);
    printf("Szurkearnyalatos beolvasva.\n");
    return grayImg;
}

img* gaussianBlurring(img *inputImg) 
{
    if (!inputImg || !inputImg->values || 
        inputImg->width < 3 || inputImg->height < 3 || inputImg->channels != 1) {
        printf("Ervenytelen bemeneti kep.\n");
        return NULL;
    }

    img* blurredImg = (img*) malloc(sizeof(img));
    if (blurredImg == NULL) {
        printf("Sikertelen memoriafoglalas Gauss elmosotthoz.\n");
        return NULL;
    }

    blurredImg->height   = inputImg->height;
    blurredImg->width    = inputImg->width;
    blurredImg->channels = inputImg->channels;
    
    const int pixels = inputImg->height * inputImg->width;
    blurredImg->values = (intensity*) malloc(pixels * sizeof(intensity));
    if (!blurredImg->values) {
        printf("Sikertelen memoriafoglalas Gauss elmosotthoz.\n");
        free(blurredImg);
        return NULL;
    }

    // Gauss szűrő szeperáltan
    //        | 1  2  1 |          | 1 |  
    // 1/16 * | 2  4  2 | => 1/4 * | 2 |  *  1/4 * | 1  2  1 | 
    //        | 1  2  1 |          | 1 | 

    // köztes eredményeknek:
    intensity *firstPassValues = (intensity*) malloc(pixels * sizeof(intensity));
    if (!firstPassValues) {
        free(blurredImg->values);
        free(blurredImg);
        return NULL;
    }

    const int width  = inputImg->width;
    const int height = inputImg->height;
    
    // 1/4 * | 1  2  1 |  elvégzése az eredeti képen
    for (int i = 0; i < height; i++) {
        for (int j = 1; j < width - 1; j++) {
            const int idx = i * width + j;

            const int left   = inputImg->values[idx - 1];
            const int mid    = inputImg->values[idx + 0];
            const int right  = inputImg->values[idx + 1];
            
            firstPassValues[idx] = (left + (mid << 1) + right) >> 2;
        }
    }

    // bal és jobb szél másolása
    for (int i = 0; i < height; i++) {
        int leftmost = i * width;
        int rightmost = leftmost + width - 1;
        firstPassValues[leftmost]  = inputImg->values[leftmost];
        firstPassValues[rightmost] = inputImg->values[rightmost];
    }

    // | 1 |
    // | 2 | * 1/4  elvégzése a köztes eredményeken
    // | 1 |
    for (int i = 1; i < height - 1; i++) {
        for (int j = 0; j < width; j++) {
            const int idx = i * width + j;

            const int up   = firstPassValues[idx - width];
            const int mid  = firstPassValues[idx + 0];
            const int down = firstPassValues[idx + width];

            blurredImg->values[idx] = (up + (mid << 1) + down) >> 2;
        }
    }

    // felső sor átmásolása a végeredménybe
    memcpy(blurredImg->values, firstPassValues, width * sizeof(intensity));

    // alsó sor átmásolása
    intensity *ptrBlurredBottom   = blurredImg->values + (height - 1) * width;
    intensity *ptrTemporaryBottom = firstPassValues         + (height - 1) * width;
    memcpy(ptrBlurredBottom, ptrTemporaryBottom, width * sizeof(intensity));


    free(firstPassValues);

    printf("Height: %d Width: %d Channels: %d. ", blurredImg->height, blurredImg->width, blurredImg->channels);
    printf("Gauss-mosott .\n");
    return blurredImg;
}

