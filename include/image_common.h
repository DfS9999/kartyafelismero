#ifndef IMAGE_COMMON_H
#define IMAGE_COMMON_H

#include <stdint.h>
#include <stdbool.h>

typedef uint8_t intensity;
typedef struct {
    int height;
    int width;
    int channels;
    intensity *values;
} img;

#endif // IMAGE_COMMON_H