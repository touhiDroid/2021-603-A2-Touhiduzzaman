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

float distance_ArrayImpl(long int size, float* a, float* b) {
    float sum = 0;

    for (int i = 0; i < size - 1; i++) {
        float diff = (a[i] - b[i]);
        sum += diff * diff;
    }

    return sum;
}

int *computeConfusionMatrix(int *predictions, ArffData *dataset) {
    int *confusionMatrix = (int *) calloc(dataset->num_classes() * dataset->num_classes(),
                                          sizeof(int)); // matrix size numberClasses x numberClasses

    for (int i = 0; i < dataset->num_instances(); i++) // for each instance compare the true class and predicted class
    {
        int trueClass = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
        int predictedClass = predictions[i];

        confusionMatrix[trueClass * dataset->num_classes() + predictedClass]++;
    }

    return confusionMatrix;
}

int *computeConfusionMatrix_ArrayImpl(int *predictions, int nClasses, int nInstances, int32 *testLastAttrsArr) {
    int *confusionMatrix = (int *) calloc(nClasses * nClasses, sizeof(int));

    for (int i = 0; i < nInstances; i++) // for each instance compare the true class and predicted class
    {
        int trueClass = testLastAttrsArr[i];
        int predictedClass = predictions[i];

        confusionMatrix[trueClass * nClasses + predictedClass]++;
    }

    return confusionMatrix;
}

float computeAccuracy(int *confusionMatrix, ArffData *dataset) {
    int successfulPredictions = 0;

    for (int i = 0; i < dataset->num_classes(); i++) {
        successfulPredictions += confusionMatrix[i * dataset->num_classes() +
                                                 i]; // elements in the diagonal are correct predictions
    }

    return successfulPredictions / (float) dataset->num_instances();
}

float computeAccuracy_ArrayImpl(int *confusionMatrix, int nClasses, long int nInstances) {
    int successfulPredictions = 0;

    for (int i = 0; i < nClasses; i++) {
        successfulPredictions += confusionMatrix[i * nClasses + i]; // elements in the diagonal are correct predictions
    }

    return successfulPredictions / ((float) nInstances * 1.0);
}

int *KNN_ArrayImpl(int trainClasses, int trainInstances, int testInstances,
                   float **trainValues, float **testValues, int32 *lastAttrs, int attrsSize, int k) {
    // Implements a sequential kNN where for each candidate query an in-place priority queue is maintained to identify the kNN's.

    // predictions is the array where you have to return the class predicted (integer) for the test dataset instances
    int *predictions = (int *) malloc(testInstances * sizeof(int));

    // stores k-NN candidates for a query vector as a sorted 2d array. First element is inner product, second is class.
    float *candidates = (float *) calloc(k * 2, sizeof(float));
    for (int i = 0; i < 2 * k; i++) {
        candidates[i] = FLT_MAX;
    }

    // Stores bincounts of each class over the final set of candidate NN
    int *classCounts = (int *) calloc(trainClasses, sizeof(int));

    for (int queryIndex = 0; queryIndex < testInstances; queryIndex++) {
        for (int keyIndex = 0; keyIndex < trainInstances; keyIndex++) {

            float dist = distance_ArrayImpl(attrsSize, testValues[queryIndex], trainValues[keyIndex]);

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
                    candidates[2 * c + 1] = lastAttrs[keyIndex]; // class value

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
        for (int i = 0; i < trainClasses; i++) {
            if (classCounts[i] > max) {
                max = classCounts[i];
                max_index = i;
            }
        }

        predictions[queryIndex] = max_index;

        for (int i = 0; i < 2 * k; i++) { candidates[i] = FLT_MAX; }
        memset(classCounts, 0, trainClasses * sizeof(int));
    }

    return predictions;
}

int *KNN(ArffData *train, ArffData *test, int k) {
    // Implements a sequential kNN where for each candidate query an in-place priority queue is maintained to identify the kNN's.

    // predictions is the array where you have to return the class predicted (integer) for the test dataset instances
    int *predictions = (int *) malloc(test->num_instances() * sizeof(int));

    // stores k-NN candidates for a query vector as a sorted 2d array. First element is inner product, second is class.
    float *candidates = (float *) calloc(k * 2, sizeof(float));
    for (int i = 0; i < 2 * k; i++) { candidates[i] = FLT_MAX; }

    int num_classes = train->num_classes();

    // Stores bincounts of each class over the final set of candidate NN
    int *classCounts = (int *) calloc(num_classes, sizeof(int));

    for (int queryIndex = 0; queryIndex < test->num_instances(); queryIndex++) {
        for (int keyIndex = 0; keyIndex < train->num_instances(); keyIndex++) {

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
                    candidates[2 * c + 1] = train->get_instance(keyIndex)->get(
                            train->num_attributes() - 1)->operator float(); // class value

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
    long int attrs_size = train->num_attributes();

    int tri = train->num_instances();
    int tsi = test->num_instances();

    float **trainValues = new float *[tri];
    float **testValues = new float *[tsi];
    for(int i=0; i<tri; i++)
        trainValues[i] = new float[attrs_size];
    for(int i=0; i<tsi; i++)
        testValues[i] = new float[attrs_size];

    int32 *testLastAttrArr =  (int32 *) malloc(test->num_instances() * sizeof(int32));
    for(int i=0; i < tsi; i++)
        testLastAttrArr[i] = test->get_instance(i)->get(test->num_attributes() - 1)->operator int32();

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    predictions = KNN(train, test, k);
    // predictions = KNN_ArrayImpl(train->num_classes(), tri, tsi, trainValues, testValues, testLastAttrArr, attrs_size, k);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    // Compute the confusion matrix
    int *confusionMatrix = computeConfusionMatrix(predictions, test);
    // Calculate the accuracy
    float accuracy = computeAccuracy(confusionMatrix, test);

    /*// Compute the confusion matrix
    int *confusionMatrix = computeConfusionMatrix_ArrayImpl(predictions, test->num_classes(),
                                                            test->num_instances(), testLastAttrArr);
    // Calculate the accuracy
    float accuracy = computeAccuracy_ArrayImpl(confusionMatrix, test->num_classes(), test->num_instances());*/

    uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;
    printf("%i-NN  -  %lu test  - %lu train  -  %llu ms  -   %.4f%%\n", k,
           test->num_instances(), train->num_instances(),
           (long long unsigned int) diff, accuracy);
    /*printf("The %i-NN classifier for %lu test instances on %lu train instances required %llu ms CPU time. "
           "Accuracy was %.4f\n", k, test->num_instances(), train->num_instances(),
           (long long unsigned int) diff, accuracy);*/

    return 0;
}
