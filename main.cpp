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
#include <omp.h>
#include <pthread.h>

#define SEQUENTIAL 1
#define THREADED 2
#define OPENMP 3

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

int *KNNOpenMP(ArffData *train, ArffData *test, int k, int num_desired_threads) {
    // omp_set_num_threads(128);
    // Implements a sequential kNN where for each candidate query an in-place priority queue is maintained to identify the kNN's.

    // predictions is the array where you have to return the class predicted (integer) for the test dataset instances
    int *predictions = (int *) malloc(test->num_instances() * sizeof(int));

    int num_classes = train->num_classes();

#pragma omp parallel for num_threads(num_desired_threads)
    for (int queryIndex = 0; queryIndex < test->num_instances(); queryIndex++) {

        // stores k-NN candidates for a query vector as a sorted 2d array. First element is inner product, second is class.
        float *candidates = (float *) calloc(k * 2, sizeof(float));
        for (int i = 0; i < 2 * k; i++) { candidates[i] = FLT_MAX; }

        // Stores bincounts of each class over the final set of candidate NN
        int *classCounts = (int *) calloc(num_classes, sizeof(int));

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


int *computeConfusionMatrix_ArrayImpl(int *predictions, int nClasses, int nInstances, int32 *lastAttributesArr) {
    int *confusionMatrix = (int *) calloc(nClasses * nClasses, sizeof(int));

    for (int i = 0; i < nInstances; i++) // for each instance compare the true class and predicted class
    {
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

int main(int argc, char *argv[]) {

    if (argc < 6) {
        cout << "Usage: ./main datasets/train.arff datasets/test.arff k NUM_DESIRED_THREADS VERSION"
                "\n\t\tk = 3 (typically)\n\t\tNUM_DESIRED_THREADS = 1,2,4,8,16,32,64 or 128"
                "\n\t\tVERSION = 1 for Sequential, 2 for Threaded, 3 for OpenMP" << endl;
        exit(0);
    }

    int k = strtol(argv[3], NULL, 10);
    int num_desired_threads = strtol(argv[4], NULL, 10);
    int version = strtol(argv[5], NULL, 10);

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

    int32 *testLastAttrsArr =  (int32 *) malloc(testInstances * sizeof(int32));
    for(int i=0; i < testInstances; i++)
        testLastAttrsArr[i] = test->get_instance(i)->get(numAttrs - 1)->operator int32();


    // region : Sequential Version
    if (version == SEQUENTIAL) {
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        predictions = KNN_ArrayImpl(nClasses, trainInstances, testInstances, numAttrs, train, test, k);
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
        // Compute the confusion matrix
        int *confusionMatrix = computeConfusionMatrix_ArrayImpl(predictions, nClasses,
                                                                testInstances, testLastAttrsArr);// (predictions, test);
        // Calculate the accuracy
        float accuracy = computeAccuracy_ArrayImpl(confusionMatrix, test->num_classes(),
                                                   test->num_instances());// (confusionMatrix, test);
        uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;
        printf("The %i-NN classifier for %lu test instances on %lu train instances required %llu ms CPU time. "
               "Accuracy was %.4f\n", k, testInstances, trainInstances,
               (long long unsigned int) diff, accuracy);
    }
        // endregion : Sequential Version

        // region : OpenMP version
    else if (version == OPENMP) {
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        predictions = KNNOpenMP(train, test, k, num_desired_threads);
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
        // Compute the confusion matrix
        int *confusionMatrix = computeConfusionMatrix(predictions, test);
        // Calculate the accuracy
        float accuracy = computeAccuracy(confusionMatrix, test);
        uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;
        printf("The %i-NN classifier for %lu test instances on %lu train instances required %llu ms CPU time. "
               "Accuracy was %.4f\n", k, test->num_instances(), train->num_instances(),
               (long long unsigned int) diff, accuracy);
    } // endregion : OpenMP Version

        // region : Threaded Version
    else if (version == THREADED) {
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        predictions = KNNThreaded(train, test, k, num_desired_threads);
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
        // Compute the confusion matrix
        int *confusionMatrix = computeConfusionMatrix(predictions, test);
        // Calculate the accuracy
        float accuracy = computeAccuracy(confusionMatrix, test);
        uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;
        printf("The %i-NN classifier for %lu test instances on %lu train instances required %llu ms CPU time. "
               "Accuracy was %.4f\n", k, test->num_instances(), train->num_instances(),
               (long long unsigned int) diff, accuracy);
    } // endregion : Threaded Version
        /*else if (version == MPI) {
            // Still a TO-DO MPI Version -> Done inside mpi.cpp
        }*/
    else {
        cout << "Invalid Version!\nVERSION ->\n\t1 for Sequential,\n\t2 for Threaded,\n\t3 for OpenMP"
             << endl;
        exit(0);
    }
}
