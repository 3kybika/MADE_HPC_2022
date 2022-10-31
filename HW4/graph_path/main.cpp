#include <omp.h>

#include "matrix.hpp"
#include "matrix_utils.hpp"

#define MATRIX_SIZE 5
#define PATH_LENGTH 3

// Compile: g++ -fopenmp -O3 main.cpp -o graph_path; ./graph_path

int main() {
    double perf_time;
    int pow = 3;

    Matrix adj_matrix(MATRIX_SIZE), res_matrix(MATRIX_SIZE);
    fill_matrix(adj_matrix);
    fill_matrix(res_matrix, 0);

    perf_time = omp_get_wtime();
    mat_power(adj_matrix, res_matrix, PATH_LENGTH);
    perf_time = omp_get_wtime() - perf_time;

    printf("For adjacency matrix:\n");
    print_matrix(adj_matrix);

    printf("Paths with lengths %d:\n", PATH_LENGTH);
    print_matrix(res_matrix);

    printf("Spent time: %5.3f\n", perf_time);

    return 0;
}