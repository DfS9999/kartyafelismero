#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/image_IO.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_WINDOWS_UTF8
#include "../lib/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBIW_WINDOWS_UTF8
#include "../lib/stb_image_write.h"

img* loadImage(const char *inputPath)
{
    img* newImg = (img*) malloc(sizeof(img));
    if (newImg == NULL) {
        printf("Memoriafoglalas sikertelen.\n");
        return NULL;
    }

    if (!stbi_info(inputPath, &newImg->height, &newImg->width, &newImg->channels)) {
        printf("%s kepadatainak beolvasasa sikertelen.\n", inputPath);
        free(newImg);
        return NULL;
    }
    
    newImg->values = stbi_load(inputPath, &newImg->width, &newImg->height, &newImg->channels, 0);
    if (newImg->values == NULL) {
        printf("%s beolvasasa sikertelen.\n", inputPath);
        free(newImg);
        return NULL;
    }

    printf("Height: %d Width: %d Channels: %d. ", newImg->height, newImg->width, newImg->channels);
    printf("%s beolvasva.\n", inputPath);
    return newImg;
}

int saveImage(img *image, const char *name)
{
    // bmp format
    char extension[] = ".bmp";

    int fullLength = strlen(name) + strlen(extension) + 1;
    char filename[fullLength];
    strcpy(filename, name);
    strcat(filename, extension);

    int result = stbi_write_bmp(filename, image->width, image->height, image->channels, image->values);
    if (!result) {
        printf("%s kiirasa sikertelen.\n", filename);
        return 1;
    }

    printf("%s mentve.\n", filename); 
    return 0;
}

int writeToText(img *image, const char *filename)
{
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        printf("Szoveges fajl letrehozasa sikertelen.\n");
        return 1;
    }

    int rows = image->height;
    int width = image->width;
    int channels = image->channels;

    // buffer a kiíráshoz: "255 " 4 karakter / színcsatorna, + extra karaktereknek
    size_t buffSize = rows * width * channels * 4 * 2;
    char *buff = (char*) malloc(buffSize);
    if (!buff) {
        fclose(fp);
        printf("Sikertelen memoriafoglalas.\n");
        return 1;
    }
    char *buffPtr = buff;

    for (int row = 0; row < rows; row++) {
        *buffPtr++ = '|';
        *buffPtr++ = ' ';
        int startOfRow = row * width * channels;

        for (int pixel = 0; pixel < width; pixel++) {

            int startOfPixel = startOfRow + pixel * channels;

            for (int offset = 0; offset < channels; offset++){
                sprintf(buffPtr, "%03d ", image->values[startOfPixel + offset]);
                buffPtr += 4;
            }

            *buffPtr++ = '|';
            *buffPtr++ = ' ';
        }

        *buffPtr++ = '\n';
    }
    *buffPtr = '\0';
    fputs(buff, fp);

    free(buff);
    fclose(fp);
    printf("%s mentve.\n", filename);
    return 0;
}
