#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#define THREADS_NUM 8
#define EPS 1e-6
#define DEBUG false

// Compile: g++ -fopenmp -O3 main.cpp -o pagerank; ./pagerank

void random_matrix(double *matrix, const size_t size) {
    srand(time(NULL));
    #pragma omp parallel num_threads(8)
    {
        unsigned int seed = 1234 + omp_get_thread_num();

        #pragma omp for
        for (size_t i = 0; i < size * size; ++i)
            matrix[i] = static_cast<size_t>(rand_r(&seed) % 2);
    }
}

void print_matrix(double *matrix, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j)
            printf("%5.3f, ", matrix[i * size + j]);

        printf("\n");
    }
}

void print_vector(double *vector, size_t size) {
    for (size_t i = 0; i < size; ++i)
        printf("%7.3f, ", vector[i]);

    printf("\n");
}

void debug_print_matrix(double *matrix, size_t size) {
    for (size_t i = 0; i < (size > 10 ? 10 : size); ++i)
        for (size_t j = 0; j < (size > 10 ? 10 : size); ++j)
            printf("%7.3f, ", matrix[i * size + j]);

    printf("\n");
}

void fill_matrix(double *matrix, size_t size, double value) {
    #pragma omp for
    for (size_t i = 0; i < size * size; ++i)
        matrix[i] = value;
}

void fill_vector(double * vector, size_t size, double value) {
    #pragma omp for
    for (size_t i = 0; i < size; ++i)
        vector[i] = value;
}

void naive_ranking(double *matrix, double* result_ranking, const size_t size) {
    fill_vector(result_ranking, size, 0);

    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        size_t offset = i * size;
        for (size_t j = 0; j < size; ++j)
            result_ranking[i] += matrix[offset + j];
    }
}

double get_l1_notm(double* vector, const size_t size) {
    double result = 0.0;

    #pragma omp parallel for reduction(+: result)
    for (size_t i = 0; i < size; ++i)
        result += vector[i];

    return result;
}

void nomalize_vector(double* vector, const size_t size, double norm) {
    #pragma omp parallel for shared(norm)
    for (size_t i = 0; i < size; ++i) {
        vector[i] /= norm;
    }
}

void matr_vec_mul(double* matrix, double* vector, double* result_vector, size_t size) {
    #pragma omp parallel for
     for (size_t i = 0; i < size; ++i) {
        result_vector[i] = 0;
        size_t offset = i * size;
        for (size_t j = 0; j < size; ++j)
            result_vector[i] += matrix[offset + j] * vector[j];
    }
}

void pagerank(double *matrix, double* ranking, const size_t size, double damping = 0.85) {

    double* result_ranking = new double[size];
    double* buffer_ranking = new double[size];

    #pragma omp parallel for 
    for (size_t i = 0; i < size; ++i)
        result_ranking[i] = 1.0 / size;

    double cur_norm = get_l1_notm(result_ranking, size);
    nomalize_vector(&result_ranking[0], size, cur_norm);

    double* new_matrix = new double[size * size];
    fill_matrix(new_matrix, size, 0);

    #pragma omp parallel for shared(damping)
    for (size_t i = 0; i < size; ++i) {
        size_t offset = i * size;
        for (size_t  j = 0; j < size; ++j) 
            new_matrix[offset + j] = damping * matrix[offset + j] + (1 - damping) / size;
    }
    
    double prev_norm = 0.0;

    while (abs(cur_norm - prev_norm) > EPS) {
        prev_norm = cur_norm;

        matr_vec_mul(new_matrix, result_ranking, buffer_ranking, size);

        auto tmp_ptr = result_ranking;
        result_ranking = buffer_ranking;
        buffer_ranking = tmp_ptr;

        cur_norm = get_l1_notm(result_ranking, size);
        nomalize_vector(result_ranking, size, cur_norm);

        if (DEBUG) {
            printf("norm = %lf\n", cur_norm);
            printf("\ncur_ranking\n");
            print_vector(result_ranking, size);
        }
    }

    #pragma omp parallel for
    for (size_t i =0; i < size; ++i)
        ranking[i] = result_ranking[i];

    delete [] result_ranking;
    delete [] buffer_ranking;
}

int main(int argc, const char* argv[]) {

    size_t size = 10;
    double perf_time;

    double* graph = new double[size * size];
    random_matrix(graph, size);

    printf("Initial matrix:\n");
    print_matrix(graph, size);

    double* ranks = new double[size];

    perf_time = omp_get_wtime();
    naive_ranking(graph, ranks, size);
    perf_time = omp_get_wtime() - perf_time;
    
    printf("\nNaive ranking:\n");
    for (size_t i = 0; i < size; ++i)
        printf("%3.zd: %lf\n", i + 1, ranks[i]);
    printf("Spent time: %5.3f\n", perf_time);

    perf_time = omp_get_wtime();
    pagerank(graph, ranks, size);
    perf_time = omp_get_wtime() - perf_time;
    printf("\nPagerank solution (percent):\n");
    for (size_t i = 0; i < size; ++i)
        printf("%3.zd: %lf\n", i + 1, ranks[i] * 100);
    printf("Spent time: %5.3f\n", perf_time);

    delete [] graph;
    delete [] ranks;
    
    return 0;
}
