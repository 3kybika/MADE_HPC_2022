#pragma once
#include <stdio.h>

struct Matrix {
    Matrix(size_t size):
        size(size),
        matrix(new double[size * size])
    {};

    ~Matrix() {
        delete [] matrix;
    }

    double* matrix;
    size_t size;
};
