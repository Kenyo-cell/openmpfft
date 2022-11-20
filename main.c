#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <omp.h>

typedef double complex cplx;
#define cx_size sizeof(cplx)

cplx *roots;
int N = 0;

cplx w_n_k(unsigned n, unsigned k) {
    return cexp(-I * M_PI * k / n);
}

cplx* _fft(cplx buf[], unsigned n) {
    if (n == 1) {
        return buf;
    }

    if (n % 2 != 0) {
        printf("array must have even capacity\n");
        return NULL;
    }

    unsigned n2 = n / 2;
    cplx *right = calloc(n2, sizeof(cplx));
    cplx *left = calloc(n2, sizeof(cplx));

    for (int i = 0; i < n2; i++) {
        *(left + i) = buf[2 * i];
        *(right + i) = buf[1 + 2 * i];
    }

    left = _fft(left, n2);
    right = _fft(right, n2);

    cplx* merged = calloc(n, sizeof(cplx));

    for (int i = 0; i < n2; i++) {
        cplx w = w_n_k(n, i * 2);
        printf("i: %d t: (%g; %g)\n", i * 2, creal(w), cimag(w));
        *(merged + i) = *(left + i) + w * *(right + i);
        *(merged + n2 + i) = *(left + i) - w * *(right + i);
    }

    free(left);
    free(right);

    return merged;
}

void show(const char * s, cplx buf[], unsigned n) {
    printf("%s", s);
    for (int i = 0; i < n; i++)
        if (!cimag(buf[i]))
            printf("%g ", creal(buf[i]));
        else
            printf("(%g, %g) ", creal(buf[i]), cimag(buf[i]));

    printf("\n");
}

unsigned int bits(unsigned int n) {
    unsigned int count = 0;
    while (n) {
        count += n & 1;
        n >>= 1;
    }
    return count;
}

unsigned rev(unsigned n) {
    unsigned reversed = 0;
    for (int i = 0; i < N; i++) {
        reversed |= ((n >> i) & 1) << (N - i - 1);
    }
    return reversed;
}

cplx* bit_reverse_copy(cplx a[], unsigned n) {
    cplx* A = calloc(n, cx_size);
    for (int k = 0; k < n; k++) {
        A[rev(k)] = a[k];
    }
    return A;
}

cplx* fft(cplx* array, unsigned n) {
    double power = log2(n);
    if (power != trunc(power)) {
        printf("array size is not power of 2\n");
        return NULL;
    }
    cplx *dft = bit_reverse_copy(array, n);
    #pragma par
    for (int s = 1; s <= power; s++) {
        int m = pow(2, s);
        cplx w_m = cexp(2 * M_PI * -I / m);
        for (int k = 0; k < n; k += m) {
            cplx w = 1;
            for (int j = 0; j < m / 2; j++) {
                cplx t = w * dft[k + j + m / 2];
                cplx u = dft[k + j];
                dft[k + j] = u + t;
                dft[k + j + m / 2] = u - t;
                w = w * w_m;
            }
        }
    }
    return dft;
}

cplx* precalculate(unsigned order) {
    cplx* complex_roots = calloc(order, sizeof(cplx));
    if (complex_roots == NULL) {
        printf("unable to allocate memory!");
    }
    for (int j = 0; j < order; j++) {
        *(complex_roots + j) = cexp(-I * 2 * M_PI * j / order);
    }
    return complex_roots;
}

int main() {
    // PI = atan2(1, 1) * 4;
    cplx buf[] = {5, 2, 4, -1, 5, 2, 4, -1, 5, 2, 4, -1, 5, 2, 4, -1 };
    unsigned n = 4;
    N = (int) ceil(log2(n));
    show("Data: ", buf, n);
    cplx* res = fft(buf, n);
    show("\nFFT : ", res, n);
    // roots = precalculate(8);
    // for (int j = 0; j < 8; j++) {
    //     printf("(%g; %g)\n", creal(*(roots + j)), cimag(*(roots + j)));
    // }

    return 0;
}