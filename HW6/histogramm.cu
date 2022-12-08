/*
Compiling:
    nvcc -arch sm_86 ./histogramm.cu -o histogramm
Using:
    ./histogramm <source_file_name>
    source_file_name - path to source file, default value: default.jpg
*/ 


#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define GRID_CNT 4

__global__ void calcHistogram(
    uint8_t* source_image, 
    double * histogram_arr, 
    int image_size, 
    int chunk_cnt
) {

    int block_start = (
        threadIdx.z * blockDim.x * blockDim.y + 
        threadIdx.y * blockDim.x + 
        threadIdx.x
    ) + (
        blockIdx.z * gridDim.x * gridDim.y +
        blockIdx.y * gridDim.x +
        blockIdx.x
    ) * blockDim.x * blockDim.y * blockDim.z;

    int chunk_image_size = image_size / chunk_cnt;
    int start_idx = chunk_image_size * block_start;
    int end_idx = start_idx + chunk_image_size;

    double localHistogram[256];
    for (int i = 0; i < 256; ++i)
        localHistogram[i] = 0.;

    if (end_idx > image_size)
        end_idx = image_size;

    for(int i = start_idx; i < end_idx; ++i)
        localHistogram[source_image[i]] += 1.;

    for(int i = 0; i < 256; ++i)
        atomicAdd(&histogram_arr[i], localHistogram[i]);

}

int main(int argc, char **argv) {

    if (argc == 2 && (
            strcmp(argv[1], "-h") == 0 ||
            strcmp(argv[1], "--help") == 0
        )
    ) {
        printf("App for getting histogram of image\n\n");
        printf("Compiling:\n");
        printf("    nvcc -arch sm_86 ./histogramm.cu -o histogramm\n");
        printf("Using:\n");
        printf("    ./median_filter <source_file_name>\n");
        printf("    source_file_name - path to source file, default value: default.jpg\n");

        return 0;
    }

    // args
    char* source_path;

    if (argc == 2)
        source_path = argv[1];
    else
        source_path = (char*) "default.jpg";

    int width;
    int height;
    int comp;
    uint8_t* source_image = stbi_load(source_path, &width, &height, &comp, 1);
    int image_size = height * width * sizeof(uint8_t);

    // cuda source image
    uint8_t* cuda_source_image;
    cudaMalloc(&cuda_source_image, image_size);
    cudaMemcpy(cuda_source_image, source_image, image_size, cudaMemcpyHostToDevice);

    // histogramm array
    double * histogram_arr = (double*) malloc(256 * sizeof(double));
    for (int i = 0; i < 256; ++i)
        histogram_arr[i] = 0.;

    // cuda histogramm array
    double * cuda_histogram_arr;
    cudaMalloc(&cuda_histogram_arr, 256 * sizeof(double));
    cudaMemcpy(cuda_histogram_arr, histogram_arr, 256 * sizeof(double), cudaMemcpyHostToDevice);

    // getting histogram
    calcHistogram<<<1, GRID_CNT * GRID_CNT>>>(
        cuda_source_image, cuda_histogram_arr, image_size, GRID_CNT * GRID_CNT
    );
    cudaDeviceSynchronize();

    cudaMemcpy(histogram_arr, cuda_histogram_arr, 256 * sizeof(double), cudaMemcpyDeviceToHost);

    // get sum of brightness levels for normalizing histogram
    double sum_brightness = 0.;
    for (int i = 0; i < 256; ++i)
        sum_brightness += histogram_arr[i];

    printf("\nResult for %s:\n", source_path);
    for (int i = 0; i < 256; ++i)
        printf("%3d: %8.8f\n", i, histogram_arr[i] / sum_brightness);

    stbi_image_free(source_image);
    free(histogram_arr); 
    cudaFree(cuda_source_image);
    cudaFree(cuda_histogram_arr);  
 
    return 0;
}
