#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_WINDOWS_UTF8
#include "stb_image.h" 
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBIW_WINDOWS_UTF8
#include "stb_image_write.h"

#define GRAYSCALER_R    0.3f
#define GRAYSCALER_G    0.59f
#define GRAYSCALER_B    0.11f
#define WIDTH           128

typedef uint8_t intensity;
typedef struct {
    int height;
    int width;
    int channels;
    intensity *values;
} img;

int loadImage(const char *inputPath, img *outImg);      // input kép betöltése
int saveImage(img *image, const char *name);            // írás bmp formátumba
int convertToGrayscale(img *coloredImg, img *outImg);   // szürkeárnyalatosítás
void writeToText(img *image, const char *filename);     // kép kiírás | XXX XXX XXX | alakban
img* templateMatchingOnGrayscale(img *subject, img *template);



int main(int argc, char *argv[])
{
    if (argc < 3) {
        //printf("Adj eleresi utat a kephez: %s <image.jpg>\n", argv[0]);
        printf("Adj eleresi utat a kephez: %s <image.jpg> <template.jpg> \n", argv[0]);
        return 1;
    }
    img original;
    if (loadImage(argv[1], &original)) {
        return 1;
    }
    img grayscaleOriginal;
    if (convertToGrayscale(&original, &grayscaleOriginal)) {
        stbi_image_free(original.values);
        return 1;
    }

    img template;
    if (loadImage(argv[2], &template)) {
        return 1;
    }
    img grayscaleTemplate;
    if (convertToGrayscale(&template, &grayscaleTemplate)) {
        stbi_image_free(template.values);
        return 1;
    }

    img *resultImagePtr = templateMatchingOnGrayscale(&grayscaleOriginal, &grayscaleTemplate);
    if (!resultImagePtr) {
        printf("Template matching sikertelen\n");
    } else {
        saveImage(resultImagePtr, "templateMatchingResult");
    }
    
    //writeToText(&original, "original");
    //writeToText(&grayscale, "grayscale");

    free(resultImagePtr->values);
    free(resultImagePtr);

    return 0;
}


int loadImage(const char *inputPath, img *outImg)
{
    if (!stbi_info(inputPath, &outImg->height, &outImg->width, &outImg->channels)) {
        printf("%s kepadatainak beolvasasa sikertelen.\n", inputPath);
        return 1;
    }
    
    outImg->values = stbi_load(inputPath, &outImg->width, &outImg->height, &outImg->channels, 0);
    if (outImg->values == NULL) {
        printf("%s beolvasasa sikertelen.\n", inputPath);
        return 1;
    }

    printf("Height: %d Width: %d Channels: %d. ", outImg->height, outImg->width, outImg->channels);
    printf("%s beolvasva.\n", inputPath);
    return 0;
}

int convertToGrayscale(img *coloredImg, img *outImg)
{
    outImg->width = coloredImg->width;
    outImg->height = coloredImg->height;
    outImg->channels = 1;

    int pixels = coloredImg->height * coloredImg->width;
    outImg->values = (intensity*) malloc(pixels * sizeof(intensity));
    if (!outImg->values) {
        printf("Sikertelen memoriafoglalas szurkearnyalatoshoz.\n");
        return 1;
    }

    int channels = coloredImg->channels;
    if (channels < 3) {
        for (int i = 0; i < pixels; ++i) {
            outImg->values[i] = coloredImg->values[i];
        }
    }
    else {
        for (int i = 0; i < pixels; ++i) {
            // R0 G0 B0 R1 G1 B1
            intensity r = coloredImg->values[i * channels + 0];
            intensity g = coloredImg->values[i * channels + 1];
            intensity b = coloredImg->values[i * channels + 2];
            outImg->values[i] = r*GRAYSCALER_R + g*GRAYSCALER_G + b*GRAYSCALER_B;
        }
    }

    printf("Height: %d Width: %d Channels: %d. ", outImg->height, outImg->width, outImg->channels);
    printf("Szurkearnyalatos beolvasva.\n");
    return 0;
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

void writeToText(img *image, const char *filename)
{
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        printf("Szoveges fajl letrehozasa sikertelen.\n");
        return;
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
        return;
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
}

img* templateMatchingOnGrayscale(img *subject, img *template)
{
    int subjectH = subject->height;
    int subjectW = subject->width;
    int templateH = template->height;
    int templateW = template->width;
    intensity *subjectVals = subject->values;
    intensity *templateVals = template->values;
    
    /////
    if (subject->channels != 1 || template->channels != 1) { 
        printf("Nem szurkearnyalatos kep input\n");
        return NULL;
    }
    if (templateH > subjectH || templateW > subjectW) {
        printf("Template nagyobb meretu mint input kep\n");
        return NULL;
    } 
    
    // különbség értékek tombje, kezdetben INT_MAX
    int pixelCount = subjectH * subjectW;
    int *results = (int*) malloc(pixelCount * sizeof(int));
    if (!results) {
        printf("Sikertelen memoriafoglalas.\n");
        return NULL;
    }
    for (int i = 0; i < pixelCount; results[i++] = INT_MAX);

    // különbségek és minimum-maximum értékek meghatározása
    int minDiff = INT_MAX;
    int maxDiff = -1;
    for (int i = 0; i <= subjectH - templateH; i++) {
        for (int j = 0; j <= subjectW - templateW; j++) {
            int diff = 0;

            for (int k = 0; k < templateH; k++) {
                for (int l = 0; l < templateW; l++) {
                    int sIdx = (i + k) * subjectW + j + l;
                    int tIdx = k * templateW + l;

                    diff += abs(subjectVals[sIdx] - templateVals[tIdx]);
                }
            }

            if (diff < minDiff) minDiff = diff;
            if (diff > maxDiff) maxDiff = diff;

            int rIdx = i * subjectW + j;
            results[rIdx] = diff;
        }
    }

    // eredmény képe
    img *visualization = (img*) malloc(sizeof(img));
    if (!visualization) {
        printf("Sikertelen memoriafoglalas.\n");
        free(results);
        return NULL;
    }

    visualization->height = subjectH;
    visualization->width = subjectW;
    visualization->channels = 1;
    visualization->values = (intensity*) malloc(subjectH * subjectW * sizeof(intensity));
    if (!visualization->values) {
        printf("Sikertelen memoriafoglalas.\n");
        free(results);
        free(visualization);
        return NULL;
    }

    // Eredmény tömb értékeinek normalizálása 0 - 255 közé
    int diffRange = maxDiff - minDiff;
    if (diffRange == 0) {
        free(results);
        free(visualization->values);
        free(visualization);
        return NULL;
    }

    for (int i = 0; i < pixelCount; i++) {
        if (results[i] == INT_MAX) {
            visualization->values[i] = (intensity)maxDiff;
            continue;
        }
        float reduced = (float) (results[i] - minDiff) * 255.0;
        visualization->values[i] = (intensity)(reduced / diffRange);
    }
    printf("Max kulonbseg: %d, Min kulonbseg: %d\n", maxDiff, minDiff);

    free(results);

    return visualization;
}
