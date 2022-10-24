#include <omp.h>
#include <iostream>

#define SEED 123

int main(int argc, char** argv) {
    const size_t LIMIT_COUNT = 1e9;
    size_t hit_cnt = 0;
    float random_x = 0, random_y = 0;

    #pragma omp parallel \
        default(none) \
        shared(LIMIT_COUNT) \
        private(random_x, random_y) \
        reduction(+ : hit_cnt) \
        num_threads(8)
    {
        unsigned int seed = SEED + omp_get_thread_num();

        #pragma omp for
        for (size_t i = 0; i < LIMIT_COUNT; ++i) {
            random_x = rand_r(&seed) / static_cast<float>(RAND_MAX);
            random_y = rand_r(&seed) / static_cast<float>(RAND_MAX);

            if ((random_x * random_x) + (random_y * random_y) < 1)
                hit_cnt +=1;
        }
    }

    float pi = (static_cast<float>(hit_cnt) / static_cast<float>(LIMIT_COUNT)) * 4;;

    std::cout << "Estimated pi:" << pi << std::endl;

    return 0;
}
