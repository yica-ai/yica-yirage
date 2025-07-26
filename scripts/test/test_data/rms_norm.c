// RMS归一化测试
#include <math.h>
void rms_norm(float *input, float *output, int size, float eps) {
    float sum_squares = 0.0f;
    for (int i = 0; i < size; i++) {
        sum_squares += input[i] * input[i];
    }
    float rms = sqrtf(sum_squares / size + eps);
    for (int i = 0; i < size; i++) {
        output[i] = input[i] / rms;
    }
}
