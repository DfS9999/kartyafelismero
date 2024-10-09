#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include "image_common.h"

img* loadImage(const char *inputPath);                  // input kép betöltése
int saveImage(img *image, const char *name);            // írás bmp formátumba
int writeToText(img *image, const char *filename);      // kép kiírás | XXX XXX XXX | alakban


#endif // IMAGE_IO_H