/*
Compiling:
    nvcc -arch sm_86 ./median_filter.cu -o median_filter
Using:
    ./median_filter <source_file_name> <target_file_name>
    source_file_name - path to source file, default value: default.jpg
    target_file_name - path to target file, default value: default_median_filter.jpg
*/ 

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define GRID_CNT 4
#define FILTER_SIZE 5 // WARNING! Must be odd!

__global__ void apply_median_filter(
    uint8_t* source_image,
    uint8_t* target_image,
    int width,
    int height,
    int size,
    int block_width,
    int block_height
) {

    int half_filter_size = FILTER_SIZE / 2;

    int block_start = (
        threadIdx.z * blockDim.x * blockDim.y + 
        threadIdx.y * blockDim.x + 
        threadIdx.x
    ) + (
        blockIdx.z * gridDim.x * gridDim.y +
        blockIdx.y * gridDim.x +
        blockIdx.x
    ) * blockDim.x * blockDim.y * blockDim.z;

    int height_block_idx =  block_start / (width / block_width);
    int width_block_idx =  block_start % (height / block_width);
    uint8_t filter_buffer[FILTER_SIZE * FILTER_SIZE];

    for (int i = 0; i < block_width; i++) {
        int x = block_width * width_block_idx + i;
        if (x > width - half_filter_size || x < half_filter_size)
            continue;

        for (int j = 0; j < block_height; j++) {
            int y = block_height* height_block_idx + j;
            if (y > height - half_filter_size || y < half_filter_size)
                continue;
 
            // fill filter
            for(int k = -half_filter_size; k <= half_filter_size; ++k) {
                for(int l = -half_filter_size; l <= half_filter_size; l++) {
                    filter_buffer[
                        half_filter_size + k + (half_filter_size + l) * FILTER_SIZE
                    ] = source_image[x + k + (y + l) * width];
                }
            }

            // apply filter
            for (int k = 0; k < FILTER_SIZE * FILTER_SIZE; ++k) {
                for (int l = 0; l < FILTER_SIZE * FILTER_SIZE - k; ++l) {
                    if (filter_buffer[l] < filter_buffer[l + 1]) {
                        uint8_t tmp = filter_buffer[l];
                        filter_buffer[l] =  filter_buffer[l + 1];
                        filter_buffer[l + 1] = tmp;
                    }
                }
            }
            target_image[x + y * width] = filter_buffer[(half_filter_size + 1) * (half_filter_size + 1)];
        }
    }
}


int main(int argc, char **argv) {

    // args
    char* source_path;
    char* target_path;

    if (argc == 2 && (
            strcmp(argv[1], "-h") == 0 ||
            strcmp(argv[1], "--help") == 0
        )
    ) {
        printf("App for blurring the image by median filter\n\n");
        printf("Compiling:\n");
        printf("    nvcc -arch sm_86 ./median_filter.cu -o median_filter\n");
        printf("Using:\n");
        printf("    ./median_filter <source_file_name> <target_file_name>\n");
        printf("    source_file_name - path to source file, default value: default.jpg\n");
        printf("    target_file_name - path to target file, default value: default_median_filter.jpg\n");

        return 0;
    }

    if (argc >= 2)
        source_path = argv[1];
    else
        source_path = (char*) "default.jpg";

    if (argc == 3)
        target_path = argv[2];
    else
        target_path = (char*) "default_median_filter.jpg";

    // source image
    int width;
    int height;
    int comp;
    uint8_t* source_image = stbi_load(source_path, &width, &height, &comp, 1);

    int image_size = height * width * sizeof(uint8_t);
    int block_width = width / GRID_CNT;
    int block_height = height / GRID_CNT;

    // cuda source image
    uint8_t* cuda_source_image;
    cudaMalloc(&cuda_source_image, image_size);
    cudaMemcpy(cuda_source_image, source_image, image_size, cudaMemcpyHostToDevice);

    // target image
    uint8_t* target_image = (uint8_t*) malloc(image_size);

    // cuda target image
    uint8_t* cuda_target_image;
    cudaMalloc(&cuda_target_image, image_size);

    // apply filter
    apply_median_filter<<<1, GRID_CNT * GRID_CNT>>>(
        cuda_source_image, cuda_target_image, width, height, image_size, block_width, block_height
    );
    cudaDeviceSynchronize();

    // output image
    cudaMemcpy(target_image, cuda_target_image, image_size, cudaMemcpyDeviceToHost);
    stbi_write_png(target_path, width, height, 1, target_image, width);

    // free
    free(target_image);
    stbi_image_free(source_image);

    cudaFree(cuda_source_image);
    cudaFree(cuda_target_image);

    return 0;
}
