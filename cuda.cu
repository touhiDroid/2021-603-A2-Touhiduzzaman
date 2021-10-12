#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <tuple>
#include <iostream>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"
#include <bits/stdc++.h>

using namespace std;

float distance(ArffInstance *a, ArffInstance *b) {
    float sum = 0;

    for (int i = 0; i < a->size() - 1; i++) {
        float diff = (a->get(i)->operator float() - b->get(i)->operator float());
        sum += diff * diff;
    }

    return sum;
}

int *computeConfusionMatrix_ArrayImpl(int *predictions, int nClasses, int nInstances, int32 *lastAttributesArr) {
    int *confusionMatrix = (int *) calloc(nClasses * nClasses, sizeof(int));

    for (int i = 0; i < nInstances; i++) { // for each instance compare the true class and predicted class
        int trueClass = lastAttributesArr[i];
        int predictedClass = predictions[i];

        confusionMatrix[trueClass * nClasses + predictedClass]++;
    }
    return confusionMatrix;
}

float computeAccuracy_ArrayImpl(int *confusionMatrix, int nClasses, int nInstances) {
    int successfulPredictions = 0;

    for (int i = 0; i < nClasses; i++) {
        successfulPredictions += confusionMatrix[i * nClasses + i]; // elements in the diagonal are correct predictions
    }

    return successfulPredictions / ((float) nInstances * 1.0);
}

int *KNN_ArrayImpl(int num_classes, int32 trainInstances, int32 testInstances, int32 numAttrs,
                   ArffData *train, ArffData *test, int k) {
    // Implements a sequential kNN where for each candidate query an in-place priority queue is maintained to identify the kNN's.

    // predictions is the array where you have to return the class predicted (integer) for the test dataset instances
    int *predictions = (int *) malloc(test->num_instances() * sizeof(int));

    // stores k-NN candidates for a query vector as a sorted 2d array. First element is inner product, second is class.
    float *candidates = (float *) calloc(k * 2, sizeof(float));
    for (int i = 0; i < 2 * k; i++) { candidates[i] = FLT_MAX; }

    // int num_classes = train->num_classes();

    // Stores bincounts of each class over the final set of candidate NN
    int *classCounts = (int *) calloc(num_classes, sizeof(int));

    for (int queryIndex = 0; queryIndex < testInstances; queryIndex++) {
        for (int keyIndex = 0; keyIndex < trainInstances; keyIndex++) {

            float dist = distance(test->get_instance(queryIndex), train->get_instance(keyIndex));

            // Add to our candidates
            for (int c = 0; c < k; c++) {
                if (dist < candidates[2 * c]) {
                    // Found a new candidate
                    // Shift previous candidates down by one
                    for (int x = k - 2; x >= c; x--) {
                        candidates[2 * x + 2] = candidates[2 * x];
                        candidates[2 * x + 3] = candidates[2 * x + 1];
                    }

                    // Set key vector as potential k NN
                    candidates[2 * c] = dist;
                    // class value
                    candidates[2 * c + 1] = train->get_instance(keyIndex)->get(numAttrs - 1)->operator float();

                    break;
                }
            }
        }

        // Bincount the candidate labels and pick the most common
        for (int i = 0; i < k; i++) {
            classCounts[(int) candidates[2 * i + 1]] += 1;
        }

        int max = -1;
        int max_index = 0;
        for (int i = 0; i < num_classes; i++) {
            if (classCounts[i] > max) {
                max = classCounts[i];
                max_index = i;
            }
        }

        predictions[queryIndex] = max_index;

        for (int i = 0; i < 2 * k; i++) { candidates[i] = FLT_MAX; }
        memset(classCounts, 0, num_classes * sizeof(int));
    }

    return predictions;
}

int main(int argc, char *argv[]) {

    if (argc != 4) {
        cout << "Usage: ./main datasets/train.arff datasets/test.arff k" << endl;
        exit(0);
    }

    int k = strtol(argv[3], NULL, 10);

    // Open the datasets
    ArffParser parserTrain(argv[1]);
    ArffParser parserTest(argv[2]);
    ArffData *train = parserTrain.parse();
    ArffData *test = parserTest.parse();

    struct timespec start, end;
    int *predictions = NULL;

    int nClasses = test->num_classes();
    int32 trainInstances = train->num_instances();
    int32 testInstances = test->num_instances();
    int32 numAttrs = test->num_attributes();

    int32 *testLastAttrsArr = (int32 *) malloc(testInstances * sizeof(int32));
    for (int i = 0; i < testInstances; i++)
        testLastAttrsArr[i] = test->get_instance(i)->get(numAttrs - 1)->operator int32();

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    predictions = KNN_ArrayImpl(nClasses, trainInstances, testInstances, numAttrs, train, test, k);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    int *confusionMatrix = computeConfusionMatrix_ArrayImpl(
            predictions, nClasses, testInstances, testLastAttrsArr);// (predictions, test);
    float accuracy = computeAccuracy_ArrayImpl(
            confusionMatrix, test->num_classes(), test->num_instances());// (confusionMatrix, test);
    uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

    printf("%i-NN  -  %lu test  - %lu train  -  %llu ms  -   %.4f%%\n", k,
           testInstances, trainInstances, (long long unsigned int) diff, accuracy);
    /*printf("The %i-NN classifier for %lu test instances on %lu train instances required %llu ms CPU time. "
           "Accuracy was %.4f\n", k, test->num_instances(), train->num_instances(),
           (long long unsigned int) diff, accuracy);*/

    return 0;
}
