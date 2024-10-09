#include "../include/image_recognition.h"

#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>


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
