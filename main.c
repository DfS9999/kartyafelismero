#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_WINDOWS_UTF8
#include "stb_image.h" 
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBIW_WINDOWS_UTF8
#include "stb_image_write.h" // Each function returns 0 on failure and non-0 on success.
//#define STB_IMAGE_RESIZE_IMPLEMENTATION
//#include "stb_image_resize2.h"

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



int main(int argc, char *argv[])
{
    if (argc < 2) {
        printf("Adj eleresi utat a kephez: %s <image.jpg>\n", argv[0]);
        return 1;
    }

    img original;
    if (loadImage(argv[1], &original)) {
        return 1;
    }

    img grayscale;
    if (convertToGrayscale(&original, &grayscale)) {
        stbi_image_free(original.values);
        return 1;
    }

    saveImage(&grayscale, "gray");




    // free
    stbi_image_free(original.values);
    free(grayscale.values);

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

    printf("Szurkearnyalatos beolvasva.\n");
    return 0;
}

int saveImage(img *image, const char *name)
{
    int len = strlen(name) + 4 + 1;
    char filename[len];
    char extension[] = ".bmp";
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

