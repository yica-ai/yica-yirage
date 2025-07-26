// 卷积操作测试
void convolution_2d(float *input, float *kernel, float *output, 
                    int input_h, int input_w, int kernel_size) {
    int output_h = input_h - kernel_size + 1;
    int output_w = input_w - kernel_size + 1;
    
    for (int oh = 0; oh < output_h; oh++) {
        for (int ow = 0; ow < output_w; ow++) {
            float sum = 0.0f;
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int ih = oh + kh;
                    int iw = ow + kw;
                    sum += input[ih * input_w + iw] * kernel[kh * kernel_size + kw];
                }
            }
            output[oh * output_w + ow] = sum;
        }
    }
}
