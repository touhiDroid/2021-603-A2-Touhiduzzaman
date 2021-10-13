//
// Created by Md Touhiduzzaman on 10/12/21.
//

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
#include <pthread.h>

using namespace std;

struct ThreadArgs {
    ArffData *train;
    ArffData *test;
    int num_classes;
    int k;
    int start;
    int end;
    int *predictions;
};

float distance(ArffInstance *a, ArffInstance *b) {
    float sum = 0;

    for (int i = 0; i < a->size() - 1; i++) {
        float diff = (a->get(i)->operator float() - b->get(i)->operator float());
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

float computeAccuracy(int *confusionMatrix, ArffData *dataset) {
    int successfulPredictions = 0;

    for (int i = 0; i < dataset->num_classes(); i++) {
        successfulPredictions += confusionMatrix[i * dataset->num_classes() +
                                                 i]; // elements in the diagonal are correct predictions
    }

    return successfulPredictions / (float) dataset->num_instances();
}

void *KnnRunSingleThread(void *arguments) {
    struct ThreadArgs *argus = (struct ThreadArgs *) arguments;
    int k = argus->k;

    for (int queryIndex = argus->start; queryIndex < argus->end; queryIndex++) {

        // stores k-NN candidates for a query vector as a sorted 2d array. First element is inner product, second is class.
        float *candidates = (float *) calloc(k * 2, sizeof(float));
        for (int i = 0; i < 2 * k; i++) { candidates[i] = FLT_MAX; }

        // Stores bincounts of each class over the final set of candidate NN
        int *classCounts = (int *) calloc(argus->num_classes, sizeof(int));

        for (int keyIndex = 0; keyIndex < argus->train->num_instances(); keyIndex++) {

            float dist = distance(argus->test->get_instance(queryIndex), argus->train->get_instance(keyIndex));

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
                    candidates[2 * c + 1] = argus->train->get_instance(keyIndex)->get(
                            argus->train->num_attributes() - 1)->operator float(); // class value

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
        for (int i = 0; i < argus->num_classes; i++) {
            if (classCounts[i] > max) {
                max = classCounts[i];
                max_index = i;
            }
        }

        argus->predictions[queryIndex] = max_index;

        for (int i = 0; i < 2 * k; i++) { candidates[i] = FLT_MAX; }
        memset(classCounts, 0, argus->num_classes * sizeof(int));
    }

    pthread_exit(0);
}

int *KNNThreaded(ArffData *train, ArffData *test, int k, int num_desired_threads) {
    // Implements a sequential kNN where for each candidate query an in-place priority queue is maintained to identify the kNN's.

    // predictions is the array where you have to return the class predicted (integer) for the test dataset instances
    int *predictions = (int *) malloc(test->num_instances() * sizeof(int));

    int num_classes = train->num_classes();

    // Create multiple threads for the outer loop
    int n = test->num_instances();
    // int partition_size = (n + num_desired_threads - 1) / num_desired_threads;
    pthread_t *threads = (pthread_t *) malloc(num_desired_threads * sizeof(pthread_t));

    // running 1 thread short to let the last one collect all the remaining data points
    for (int t = 0; t < num_desired_threads; t++) {
        int start = t * n / num_desired_threads;
        int end = (t + 1) * n / num_desired_threads;
        if (t == num_desired_threads - 1)
            end = n;
        // call KnnRunOneThread
        struct ThreadArgs argus;
        argus.train = train;
        argus.test = test;
        argus.num_classes = num_classes;
        argus.k = k;
        argus.start = start;
        argus.end = end;
        argus.predictions = predictions;
        pthread_create(&threads[t], NULL, KnnRunSingleThread, (void *) &argus);
    }
    for (int t = 0; t < num_desired_threads; t++) {
        pthread_join(threads[t], NULL);
    }

    return predictions;
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        cout << "Usage: ./multithreaded.o datasets/train.arff datasets/test.arff k NUM_DESIRED_THREADS"
                "\n\t\tk = 3 (typically)\n\t\tNUM_DESIRED_THREADS = 1,2,4,8,16,32,64 or 128" << endl;
        exit(0);
    }

    int k = strtol(argv[3], NULL, 10);
    int num_desired_threads = strtol(argv[4], NULL, 10);

    // Open the datasets
    ArffParser parserTrain(argv[1]);
    ArffParser parserTest(argv[2]);
    ArffData *train = parserTrain.parse();
    ArffData *test = parserTest.parse();

    struct timespec start, end;
    int *predictions = NULL;

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    predictions = KNNThreaded(train, test, k, num_desired_threads);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    int *confusionMatrix = computeConfusionMatrix(predictions, test);
    float accuracy = computeAccuracy(confusionMatrix, test);

    uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

    printf("%i-NN  -  %lu test  - %lu train  -  %llu ms  -   %.4f%%\n", k,
           test->num_instances(), train->num_instances(), (long long unsigned int) diff, accuracy);

    return 0;
}
