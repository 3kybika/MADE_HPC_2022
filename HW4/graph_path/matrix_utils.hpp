#include <omp.h>
#include <iostream>
#include <iostream>
#include <exception>

#include "matrix.hpp"

#define THREADS_NUM 8
#define DEBUG false

void print_matrix(Matrix &matrix) {
    for (size_t i = 0; i < matrix.size; ++i) {
        for (size_t j = 0; j < matrix.size; ++j)
            printf("%7.3f, ", matrix.matrix[i * matrix.size + j]);
        printf("\n");
    }
}

void debug_print_matrix(Matrix &matrix) {
    size_t print_limit = matrix.size;

    for (size_t i = 0; i < print_limit; ++i) {
        for (size_t j = 0; j < print_limit; ++j)
            printf("%7.3f, ", matrix.matrix[i * matrix.size + j]);
        printf("\n");
    }
}

void fill_matrix(Matrix &matrix) {
    #pragma omp parallel num_threads(8)
    {
        unsigned int seed = 123 + omp_get_thread_num();

        #pragma omp for
        for (size_t i = 0; i < matrix.size * matrix.size; ++i)
            matrix.matrix[i] = static_cast<size_t>(rand_r(&seed) % 2);
    }
}

void fill_matrix(Matrix &matrix, double value) {
    #pragma omp for
    for (size_t i = 0; i < matrix.size * matrix.size; ++i)
        matrix.matrix[i] = value;

}

void matrix_mul(const Matrix &left_matrix, const Matrix &right_matrix, Matrix &res_matrix) {
    if (left_matrix.size != right_matrix.size) {
        throw std::runtime_error("Size error");
    }

    for (int i = 0; i < right_matrix.size; ++i)
        for(int j = 0; j < right_matrix.size; ++j) {
            res_matrix.matrix[i * right_matrix.size + j] = 0;
            for(int k = 0; k < right_matrix.size; ++k) {
                res_matrix.matrix[i * right_matrix.size + j] +=
                    left_matrix.matrix[i * right_matrix.size + k] *
                    right_matrix.matrix[k * right_matrix.size + j];
            }
        }
}

void mat_power(const Matrix& adj_matrix, Matrix& res_matrix, size_t power) {

    if (power == 0) {
        #pragma omp parallel for shared(adj_matrix, res_matrix)
        for (int i = 0; i < adj_matrix.size; ++i)
            for (int j = 0; j < adj_matrix.size; ++j)
                res_matrix.matrix[i * adj_matrix.size + j] = 0;

        #pragma omp parallel for shared(adj_matrix, res_matrix)
        for (int i = 0; i < adj_matrix.size; ++i)
            res_matrix.matrix[i * adj_matrix.size + i] = 1;

        return;
    }

    if (power == 1) {
        #pragma omp parallel for shared(adj_matrix, res_matrix)
        for (int i = 0; i < adj_matrix.size; ++i)
            for (int j = 0; j < adj_matrix.size; ++j)
                res_matrix.matrix[i * adj_matrix.size + j] = adj_matrix.matrix[i * adj_matrix.size + j];

        return;
    }

    Matrix temp_matrix(adj_matrix.size);

    if (power % 2 == 0) {
        mat_power(adj_matrix, temp_matrix, power / 2);
        matrix_mul(temp_matrix, temp_matrix, res_matrix);
    } else {
        mat_power(adj_matrix, temp_matrix, power - 1);
        matrix_mul(adj_matrix, temp_matrix, res_matrix);
    }

    if (DEBUG) {
        printf("\n\npower: %zd\n", power);
        printf("temp_matrix:\n");
        print_matrix(temp_matrix);
        printf("res_matrix:\n");
        print_matrix(res_matrix);
    }
}
