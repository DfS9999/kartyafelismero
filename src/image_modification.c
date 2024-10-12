#include "../include/image_modification.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h> // memcpy
#include <math.h>   // sqrt

//#include <omp.h> // OpenMP párhuzamosításhoz

// segéd függvények
intensity* apply1DFilter(img *inputGrayImg, int *kernel, int kernelSize, float normalizer, bool horizontal, bool copyEdge);
int* apply1DFilter_I(img *inputGrayImg, int *kernel, int kernelSize, float normalizer, bool horizontal, bool copyEdge);
int* sobelEdgeHorizontal(img *inputImg);     // Sobel él detektálás (szürkeárnyalatos input)
int* sobelEdgeVertical(img *inputImg);       // Sobel él detektálás (szürkeárnyalatos input)

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

img* gaussianBlurring3x3(img *inputImg) 
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
    blurredImg->channels = 1;
    blurredImg->values   = NULL;
    
    // Gauss szűrő szeperáltan
    //        | 1  2  1 |          | 1 |  
    // 1/16 * | 2  4  2 | => 1/4 * | 2 |  *  1/4 * | 1  2  1 | 
    //        | 1  2  1 |          | 1 | 
    int kernel[] = { 1, 2, 1 };
    int kSize = 3;
    float n = 0.25f;

    // köztes eredmény tárolásához
    img temporaryImg = { inputImg->height, inputImg->width, 1, NULL };

    // horizontálisan
    temporaryImg.values = apply1DFilter(inputImg, kernel, kSize, n, true, true);
    if (!temporaryImg.values) {
        printf("Sikertelen memoriafoglalas Gauss elmosotthoz.\n");
        free(blurredImg);
        return NULL;
    }

    // vertikális
    blurredImg->values = apply1DFilter(&temporaryImg, kernel, kSize, n, false, true);
    if (!blurredImg->values) {
        printf("Sikertelen memoriafoglalas Gauss elmosotthoz.\n");
        free(temporaryImg.values);
        free(blurredImg);
        return NULL;
    }

    printf("Height: %d Width: %d Channels: %d. ", blurredImg->height, blurredImg->width, blurredImg->channels);
    printf("Gauss-mosott .\n");

    free(temporaryImg.values);

    return blurredImg;
}

img* gaussianBlurring5x5(img *inputImg)
{
    if (!inputImg || !inputImg->values || 
        inputImg->width < 5 || inputImg->height < 5 || inputImg->channels != 1) {
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
    blurredImg->channels = 1;
    blurredImg->values   = NULL;
    
    // Gauss szűrő szeperáltan
    //         1   4   6   4  1     | 1 |
    //         4  16  24  16  4     | 4 |
    // 1/256 * 6  24  36  24  6  => | 6 | * 1/16  *  | 1  4  6  4  1 | *  1/16
    //         4  16  24  16  4     | 4 |
    //         1   4   6   4  1     | 1 |
    int kernel[] = { 1, 4, 6, 4, 1 };
    int kSize = 5;
    float n = 0.0625f;

    // köztes eredmény tárolásához
    img temporaryImg = { inputImg->height, inputImg->width, 1, NULL };

    // horizontálisan
    temporaryImg.values = apply1DFilter(inputImg, kernel, kSize, n, true, true);
    if (!temporaryImg.values) {
        printf("Sikertelen memoriafoglalas Gauss elmosotthoz.\n");
        free(blurredImg);
        return NULL;
    }

    // vertikális
    blurredImg->values = apply1DFilter(&temporaryImg, kernel, kSize, n, false, true);
    if (!blurredImg->values) {
        printf("Sikertelen memoriafoglalas Gauss elmosotthoz.\n");
        free(temporaryImg.values);
        free(blurredImg);
        return NULL;
    }

    printf("Height: %d Width: %d Channels: %d. ", blurredImg->height, blurredImg->width, blurredImg->channels);
    printf("Gauss-mosott .\n");

    free(temporaryImg.values);

    return blurredImg;
}

img* downsamplingBy2(img *inputImg)
{
    if (!inputImg || !inputImg->values || 
        inputImg->width < 3 || inputImg->height < 3 || inputImg->channels != 1) {
        printf("Ervenytelen bemeneti kep.\n");
        return NULL;
    }

    img* halvedImg = (img*) malloc(sizeof(img));
    if (halvedImg == NULL) {
        printf("Sikertelen memoriafoglalas csokkentetthez.\n");
        return NULL;
    }

    const int originalHeight = inputImg->height;
    const int originalWidth = inputImg->width;
    
    const int halvedHeight = originalHeight / 2;
    const int halvedWidth = originalWidth   / 2;
    const int pixels = halvedHeight * halvedWidth;

    halvedImg->height   = halvedHeight;
    halvedImg->width    = halvedWidth;
    halvedImg->channels = 1;
    halvedImg->values   = (intensity*) malloc(pixels * sizeof(intensity));
    if (halvedImg->values == NULL) {
        printf("Sikertelen memoriafoglalas csokkentetthez.\n");
        free(halvedImg);
        return NULL;
    }

    for (int i = 0; i < halvedHeight; i++) {
            for (int j = 0; j < halvedWidth; j++) {
                int halvedIdx   = i * halvedWidth + j;
                int originalIdx = (i * 2) * originalWidth + (j * 2);
                
                halvedImg->values[halvedIdx] = inputImg->values[originalIdx];

            }
        }

    return halvedImg;
}

img* sobelEdgeDetection(img *inputImg)
{
    if (!inputImg || !inputImg->values || 
        inputImg->width < 3 || inputImg->height < 3 || inputImg->channels != 1) {
        printf("Ervenytelen bemeneti kep.\n");
        return NULL;
    }

    img* sobelImg = (img*) malloc(sizeof(img));
    if (!sobelImg) {
        printf("Sikertelen memoriafoglalas.\n");
        return NULL;
    }

    sobelImg->height   = inputImg->height;
    sobelImg->width    = inputImg->width;
    sobelImg->channels = 1;
    sobelImg->values   = (intensity*) malloc(sobelImg->height * sobelImg->width * sizeof(intensity));
    if (!sobelImg->values) {
        printf("Sikertelen memoriafoglalas.\n");
        return NULL;
    }

    int *horizontalValues = sobelEdgeHorizontal(inputImg);
    int *verticalValues = sobelEdgeVertical(inputImg);

    // sqrt(h^2 + v^2)
    for (int i = 0; i < sobelImg->height * sobelImg->width; i++) {
        int h = horizontalValues[i];
        int v = verticalValues[i];
        double res = sqrt(h*h + v*v);

        sobelImg->values[i] = (intensity) res;
    }

    free(horizontalValues);
    free(verticalValues);

    return sobelImg;
}


//////
intensity* apply1DFilter(img *inputGrayImg, int *kernel, int kernelSize, float normalizer, bool horizontal, bool copyEdge)
{
    if (!inputGrayImg || !inputGrayImg->values || !kernel || inputGrayImg->channels != 1) {
        printf("Ervenytelen bemeneti kep.\n");
        return NULL;
    }
    if ((horizontal && kernelSize > inputGrayImg->width) || 
       (!horizontal && kernelSize > inputGrayImg->height)) {
        printf("Ervenytelen bemeneti kep.\n");
        return NULL;
    }

    intensity *srcValues      = inputGrayImg->values;
    const int width           = inputGrayImg->width;
    const int height          = inputGrayImg->height;
    intensity* filteredValues = (intensity*) malloc(height * width * sizeof(intensity));
    if (!filteredValues) {
        printf("Sikertelen memoriafoglalas.\n");
        return NULL;
    }
    
    const int side = kernelSize / 2;
    if (horizontal) {
        for (int i = 0; i < height; i++) {
            for (int j = side; j < width - side; j++) {
                const int idx = i * width + j;
                const int startIdx = idx - side;
                int sum = 0;
                for (int k = 0; k < kernelSize; k++) {
                    sum += srcValues[startIdx + k] * kernel[k];
                }

                filteredValues[idx] = (intensity)(sum * normalizer);
            }
        }
        if (copyEdge) {
            for (int i = 0; i < height; i++) {
                const int startIdx = i * width;
                // bal szélek
                for (int j = 0; j < side; j++)
                    filteredValues[startIdx + j]  = srcValues[startIdx + j];
                // jobb szélek
                for (int j = 0; j < side; j++)
                    filteredValues[startIdx + width - j] = srcValues[startIdx + width - j];
            }
        }
        // szélek nullázása
        else {
            for (int i = 0; i < height; i++) {
                const int startIdx = i * width;
                // bal szélek
                for (int j = 0; j < side; j++)
                    filteredValues[startIdx + j] = 0;
                // jobb szélek
                for (int j = 0; j < side; j++)
                    filteredValues[startIdx + width - j] = 0;
            }
        }
    }
    // vertikális
    else {
        for (int i = side; i < height - side; i++) {
            for (int j = 0; j < width; j++) {
                const int idx = i * width + j;
                const int startIdx = (i - side) * width + j;
                int sum = 0;
                for (int k = 0; k < kernelSize; k++) {
                    sum += srcValues[startIdx + (k * width)] * kernel[k];
                }

                filteredValues[idx] = (intensity)(sum * normalizer);
            }
        }
        intensity *ptrDestBottom = filteredValues + (height - side) * width;
        if (copyEdge) {
            // felső sorok
            memcpy(filteredValues, srcValues, width * side * sizeof(intensity));
            // alsó sorok
            intensity *ptrSrc  = srcValues + (height - side) * width;
            memcpy(ptrDestBottom, ptrSrc, width * side * sizeof(intensity));
        }
        else {
            // felső sorok
            memset(filteredValues, 0, side * width * sizeof(intensity));
            // alsó sorok
            memset(ptrDestBottom, 0, side * width * sizeof(intensity));
        }
    }

    return filteredValues;
}

int* apply1DFilter_I(img *inputGrayImg, int *kernel, int kernelSize, float normalizer, bool horizontal, bool copyEdge)
{
    if (!inputGrayImg || !inputGrayImg->values || !kernel || inputGrayImg->channels != 1) {
        printf("Ervenytelen bemeneti kep.\n");
        return NULL;
    }
    if ((horizontal && kernelSize > inputGrayImg->width) || 
       (!horizontal && kernelSize > inputGrayImg->height)) {
        printf("Ervenytelen bemeneti kep.\n");
        return NULL;
    }

    intensity *srcValues      = inputGrayImg->values;
    const int width           = inputGrayImg->width;
    const int height          = inputGrayImg->height;
    int* filteredValues = (int*) malloc(height * width * sizeof(int));
    if (!filteredValues) {
        printf("Sikertelen memoriafoglalas.\n");
        return NULL;
    }
    
    const int side = kernelSize / 2;
    if (horizontal) {
        for (int i = 0; i < height; i++) {
            for (int j = side; j < width - side; j++) {
                const int idx = i * width + j;
                const int startIdx = idx - side;
                int sum = 0;
                for (int k = 0; k < kernelSize; k++) {
                    sum += srcValues[startIdx + k] * kernel[k];
                }

                filteredValues[idx] = (int)(sum * normalizer);
            }
        }
        if (copyEdge) {
            for (int i = 0; i < height; i++) {
                const int startIdx = i * width;
                // bal szélek
                for (int j = 0; j < side; j++)
                    filteredValues[startIdx + j]  = srcValues[startIdx + j];
                // jobb szélek
                for (int j = 0; j < side; j++)
                    filteredValues[startIdx + width - j] = srcValues[startIdx + width - j];
            }
        }
        // szélek nullázása
        else {
            for (int i = 0; i < height; i++) {
                const int startIdx = i * width;
                // bal szélek
                for (int j = 0; j < side; j++)
                    filteredValues[startIdx + j] = 0;
                // jobb szélek
                for (int j = 0; j < side; j++)
                    filteredValues[startIdx + width - j] = 0;
            }
        }
    }
    // vertikális
    else {
        for (int i = side; i < height - side; i++) {
            for (int j = 0; j < width; j++) {
                const int idx = i * width + j;
                const int startIdx = (i - side) * width + j;
                int sum = 0;
                for (int k = 0; k < kernelSize; k++) {
                    sum += srcValues[startIdx + (k * width)] * kernel[k];
                }

                filteredValues[idx] = (int)(sum * normalizer);
            }
        }
        int *ptrDestBottom = filteredValues + (height - side) * width;
        if (copyEdge) {
            // felső sorok
            for (int i = 0; i < width * side; ++i) {
                filteredValues[i] = srcValues[i];
            }
            
            // alsó sorok
            intensity *ptrSrcBottom = srcValues + (height - side) * width;
            for (int i = 0; i < width * side; ++i) {
                ptrDestBottom[i] = ptrSrcBottom[i];
            }
        }
        else {
            // felső sorok
            memset(filteredValues, 0, side * width * sizeof(int));
            // alsó sorok
            memset(ptrDestBottom, 0, side * width * sizeof(int));
        }
    }

    return filteredValues;
}

int* sobelEdgeHorizontal(img *inputImg)
{
    if (!inputImg || !inputImg->values || 
        inputImg->width < 3 || inputImg->height < 3 || inputImg->channels != 1) {
        printf("Ervenytelen bemeneti kep.\n");
        return NULL;
    }

    // köztes eredmény tárolásához
    img temporaryImg = { inputImg->height, inputImg->width, 1, NULL };
    
    // | -1   0   1 |            | 1 |  
    // | -2   0   2 |  * 1/8  => | 2 | * 1/4  * | -1  0  1 | * 1/2
    // | -1   0   1 |            | 1 |
    int kernelV[] = { 1, 2, 1 };
    int kSize = 3;
    float nV = 0.25f;

    // 1) vertikális (uint8_t -> int)
    temporaryImg.values = apply1DFilter(inputImg, kernelV, kSize, nV, false, false);
    if (!temporaryImg.values) {
        printf("Sikertelen memoriafoglalas Gauss elmosotthoz.\n");
        return NULL;
    }

    // 2) horizontális (uint8_t -> uint8_t)
    int kernelH[] = { -1, 0, 1 };
    float nH = 0.5f;
    int *sobelValues = apply1DFilter_I(&temporaryImg, kernelH, kSize, nH, true, false);
    if (!sobelValues) {
        printf("Sikertelen memoriafoglalas Sobel él detektáláshoz.\n");
        return NULL;
    }

    free(temporaryImg.values);

    return sobelValues;
}

int* sobelEdgeVertical(img *inputImg)
{
    if (!inputImg || !inputImg->values || 
        inputImg->width < 3 || inputImg->height < 3 || inputImg->channels != 1) {
        printf("Ervenytelen bemeneti kep.\n");
        return NULL;
    }

    // köztes eredmény tárolásához
    img temporaryImg = { inputImg->height, inputImg->width, 1, NULL };
    
    // | -1  -2  -1 |           | -1 |
    // |  0   0   0 | * 1/8  => |  0 | * 1/2  * | 1  2  1 | * 1/4
    // |  1   2   1 |           |  1 |
    int kernelH[] = { 1, 2, 1 };
    int kSize = 3;
    float nH = 0.25f;

    // 1) horizontális (uint8_t -> uint8_t)
    temporaryImg.values = apply1DFilter(inputImg, kernelH, kSize, nH, true, false);
    if (!temporaryImg.values) {
        printf("Sikertelen memoriafoglalas Gauss elmosotthoz.\n");
        return NULL;
    }

    // 2) vertikális (uint8_t -> int)
    int kernelV[] = { -1, 0, 1 };
    float nV = 0.5f;
    int *sobelValues = apply1DFilter_I(&temporaryImg, kernelV, kSize, nV, false, false);
    if (!sobelValues) {
        printf("Sikertelen memoriafoglalas Sobel él detektáláshoz.\n");
        return NULL;
    }

    free(temporaryImg.values);

    return sobelValues;
}

