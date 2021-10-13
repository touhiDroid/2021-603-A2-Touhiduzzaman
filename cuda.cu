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
#include <cuda_runtime.h>

using namespace std;

int *computeConfusionMatrix(int *predictions, int nClasses, int nInstances, int32 *lastAttributesArr) {
    int *confusionMatrix = (int *) calloc(nClasses * nClasses, sizeof(int));

    for (int i = 0; i < nInstances; i++) { // for each instance compare the true class and predicted class
        int trueClass = lastAttributesArr[i];
        int predictedClass = predictions[i];

        confusionMatrix[trueClass * nClasses + predictedClass]++;
    }
    return confusionMatrix;
}

float computeAccuracy(int *confusionMatrix, int nClasses, int nInstances) {
    int successfulPredictions = 0;

    for (int i = 0; i < nClasses; i++) {
        successfulPredictions += confusionMatrix[i * nClasses + i]; // elements in the diagonal are correct predictions
    }

    return successfulPredictions / ((float) nInstances * 1.0);
}

__device__ float distance_Cuda(int32 size, float* a, int queryIndex, float* b, int keyIndex) {
    float sum = 0;

    for (int i = 0; i < size; i++) {
        float diff = (a[i + size * queryIndex] - b[i + size * keyIndex]);
        sum += diff * diff;
    }

    return sum;
}

__global__ void Kernel (int32 testInstances, int32 trainInstances, int num_classes,
                        float *trainArr, float *testArr, int32 numAttrs, int k, float *candidates, int *predictions, int *classCounts)
{
    int queryIndex = blockIdx.x*blockDim.x + threadIdx.x;
    if (queryIndex < testInstances)
    {
        int candidate_startPoint = queryIndex * 2 * k;
        int classCount_startPoint = queryIndex * num_classes;

        for (int keyIndex = 0; keyIndex < trainInstances; keyIndex++) {
            float dist = distance_Cuda(numAttrs, testArr, queryIndex, trainArr, keyIndex);

            // Add to our candidates
            for (int c = 0; c < k; c++) {
                if (dist < candidates[candidate_startPoint + 2 * c]) {
                    // Found a new candidate
                    // Shift previous candidates down by one
                    for (int x = k - 2; x >= c; x--) {
                        candidates[candidate_startPoint + 2 * x + 2] = candidates[candidate_startPoint + 2 * x];
                        candidates[candidate_startPoint + 2 * x + 3] = candidates[candidate_startPoint + 2 * x + 1];
                    }

                    // Set key vector as potential k NN
                    candidates[candidate_startPoint + 2 * c] = dist;
                    // class value
                    // candidates[2 * c + 1] = train->get_instance(keyIndex)->get(numAttrs - 1)->operator float();
                    candidates[candidate_startPoint + 2 * c + 1] = trainArr[keyIndex * numAttrs + numAttrs - 1];

                    break;
                }
            }
        }

        for (int i = 0; i < k; i++) {
            classCounts[classCount_startPoint + (int) candidates[candidate_startPoint + 2 * i + 1]] += 1;
        }

        int max = -1;
        int max_index = 0;
        for (int i = 0; i < num_classes; i++) {
            if (classCounts[classCount_startPoint + i] > max) {
                max = classCounts[classCount_startPoint + i];
                max_index = i;
            }
        }

        predictions[queryIndex] = max_index;
    }
}

int *KNN_Cuda(int num_classes, int32 trainInstances, int32 testInstances, int32 numAttrs,
              float **trainArr, float **testArr, int k, int nThreads) {
    printf("inside KNN_Cuda\n");
    // Implements a sequential kNN where for each candidate query an in-place priority queue is maintained to identify the kNN's.

    // predictions is the array where you have to return the class predicted (integer) for the test dataset instances
    int *predictions = (int *) malloc(testInstances * sizeof(int));
    printf("predictions allocated\n");

    // stores k-NN candidates for a query vector as a sorted 2d array. First element is inner product, second is class.
    float *candidates = (float *) calloc(k * 2 * testInstances, sizeof(float));
    for (int i = 0; i < 2 * k * testInstances; i++) { candidates[i] = FLT_MAX; }
    printf("candidates calloc & init with %f\n", FLT_MAX);

    // Stores bincounts of each class over the final set of candidate NN
    int *classCounts = (int *) calloc(num_classes * testInstances, sizeof(int));
    printf("classCounts calloc done\n");

    int *device_predictions;
    int *device_classCounts;
    float *device_trainArr;
    float *device_testArr;
    float *device_candidates;

    printf("cudaMalloc starting\n");
    cudaMalloc((void**)&device_predictions, testInstances * sizeof(int));
    cudaMalloc((void**)&device_classCounts,  num_classes * testInstances * sizeof(int));
    cudaMalloc((void**)&device_trainArr, trainInstances * numAttrs * sizeof(float));
    cudaMalloc((void**)&device_testArr, testInstances * numAttrs * sizeof(float));
    cudaMalloc((void**)&device_candidates, k * 2 * testInstances * sizeof(float));
    printf("cudaMalloc done\n");

    printf("cudaMemcpy starting\n");
    // copying prediction array to avoid garbage value in the device-memory
    cudaMemcpy(device_predictions, predictions, testInstances * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_classCounts, classCounts, num_classes * testInstances * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_trainArr, trainArr[0], trainInstances * numAttrs * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_testArr, testArr[0], testInstances * numAttrs * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_candidates, candidates, k * 2 * testInstances * sizeof(float), cudaMemcpyHostToDevice);
    printf("cudaMemcpy done\n");

    int THREAD_BLOCK = nThreads;

    printf("calling Kernel for chunk of testInstances=%ld with threads=%d\n", testInstances, THREAD_BLOCK);
    Kernel <<< (testInstances + THREAD_BLOCK - 1) / THREAD_BLOCK, THREAD_BLOCK >>> (testInstances, trainInstances, num_classes,
                                                                                    device_trainArr, device_testArr, numAttrs, k, device_candidates, device_predictions, device_classCounts);
    printf("done Kernel\n");

    cudaMemcpy(predictions, device_predictions, testInstances * sizeof(int), cudaMemcpyDeviceToHost);
    printf("cudaMemcpy of predictions : DONE\n\n");

    return predictions;
}


int main(int argc, char *argv[]) {
    if (argc != 5) {
        cout << "Usage: ./cuda.o datasets/train.arff datasets/test.arff k NUM_THREADS" << endl;
        exit(0);
    }

    int k = strtol(argv[3], NULL, 10);
    int nThreads = strtol(argv[4], NULL, 10);

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

    float **trainArr = new float *[trainInstances];
    for (int i = 0; i < trainInstances; i++) {
        trainArr[i] = (float *) malloc(numAttrs * sizeof(float));
        for (int j = 0; j < numAttrs; j++)
            trainArr[i][j] = train->get_instance(i)->get(j)->operator float();
    }
    printf("trainArr copied\n");
    float **testArr = new float *[testInstances];
    for (int i = 0; i < testInstances; i++) {
        testArr[i] = (float *) malloc(numAttrs * sizeof(float));
        for (int j = 0; j < numAttrs; j++)
            testArr[i][j] = test->get_instance(i)->get(j)->operator float();
    }
    printf("testArr copied\n");

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    predictions = KNN_Cuda(nClasses, trainInstances, testInstances, numAttrs, trainArr, testArr, k, nThreads);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    int *confusionMatrix = computeConfusionMatrix(
            predictions, nClasses, testInstances, testLastAttrsArr);// (predictions, test);
    float accuracy = computeAccuracy(
            confusionMatrix, test->num_classes(), test->num_instances());// (confusionMatrix, test);
    uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

    printf("\t\t%i-NN  -  %lu test  - %lu train  -  %llu ms  -   %.4f%%\n\n\n", k,
           testInstances, trainInstances, (long long unsigned int) diff, accuracy);
    /*printf("The %i-NN classifier for %lu test instances on %lu train instances required %llu ms CPU time. "
           "Accuracy was %.4f\n", k, test->num_instances(), train->num_instances(),
           (long long unsigned int) diff, accuracy);*/

    return 0;
}
