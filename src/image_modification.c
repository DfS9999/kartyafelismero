#include <stdlib.h>
#include <stdio.h>

#include "../include/image_modification.h"

#define GRAYSCALER_R    0.3f
#define GRAYSCALER_G    0.59f
#define GRAYSCALER_B    0.11f


img* convertToGrayscale(img *coloredImg)
{
    img* grayImg = (img*) malloc(sizeof(img));
    if (grayImg == NULL) {
        printf("Memoriafoglalas sikertelen.\n");
        return NULL;
    }

    grayImg->width = coloredImg->width;
    grayImg->height = coloredImg->height;
    grayImg->channels = 1;

    int pixels = coloredImg->height * coloredImg->width;
    grayImg->values = (intensity*) malloc(pixels * sizeof(intensity));
    if (!grayImg->values) {
        printf("Sikertelen memoriafoglalas szurkearnyalatoshoz.\n");
        free(grayImg);
        return NULL;
    }

    int channels = coloredImg->channels;
    if (channels < 3) {
        for (int i = 0; i < pixels; ++i) {
            grayImg->values[i] = coloredImg->values[i];
        }
    }
    else {
        for (int i = 0; i < pixels; ++i) {
            // R0 G0 B0 R1 G1 B1
            intensity r = coloredImg->values[i * channels + 0];
            intensity g = coloredImg->values[i * channels + 1];
            intensity b = coloredImg->values[i * channels + 2];
            grayImg->values[i] = r*GRAYSCALER_R + g*GRAYSCALER_G + b*GRAYSCALER_B;
        }
    }

    printf("Height: %d Width: %d Channels: %d. ", grayImg->height, grayImg->width, grayImg->channels);
    printf("Szurkearnyalatos beolvasva.\n");
    return grayImg;
}
