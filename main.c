#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <omp.h>

typedef double complex cplx;
#define cx_size sizeof(cplx)
#define true 1
#define false 0

void show(const char *s, cplx buf[], unsigned n) {
    printf("%s", s);
    for (int i = 0; i < n; i++)
        if (!cimag(buf[i]))
            printf("%g ", creal(buf[i]));
        else
            printf("(%g, %g) ", creal(buf[i]), cimag(buf[i]));

    printf("\n");
}

unsigned _rev(unsigned n, unsigned N) {
    unsigned reversed = 0;
    for (int i = 0; i < N; i++) {
        reversed |= ((n >> i) & 1) << (N - i - 1);
    }
    return reversed;
}

unsigned rev(unsigned n) {
    unsigned N = (int) ceil(log2(n));
    return _rev(n, N);
}

cplx* bit_reverse_copy(cplx a[], unsigned n) {
    cplx* A = calloc(n, cx_size);
	int k;
	#pragma omp parallel for shared(A, a, n) private(k)
    for (k = 0; k < n; k++) {
        A[rev(k)] = a[k];
    }
    return A;
}

int _power(unsigned n) {
    double power = log2(n);
    
    if (power != trunc(power)) {
        printf("array size is not power of 2\n");
        exit(1);
    }

    return (int)power;
}

cplx* _fft(cplx* array, unsigned n, short invert) {
    int power = _power(n);
    double inverted = invert == true ? -1 : 1;
    cplx *dft = bit_reverse_copy(array, n);

    for (int s = 1; s <= (int)power; s++) {
        int m = pow(2, s);
        cplx w_m = cexp(2 * M_PI * -I * inverted / m);
		
		#pragma omp parallel for shared(dft, s)
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

    if (invert) {
        #pragma omp parallel for shared(dft)
        for (int i = 0; i < n; i++) dft[i] /= n;
    }

    return dft;
}

cplx* fft(cplx* array, unsigned n) {
    return _fft(array, n, false);
}

cplx* ifft(cplx* array, unsigned n) {
    return _fft(array, n, true);
}

int main() {
    unsigned n = pow(2, 27);
    cplx *buf = calloc(n, cx_size);
	for (int i = 0; i < n; i += 4) {
		buf[i] = 5;
		buf[i + 1] = 2;
		buf[i + 2] = 4;
		buf[i + 3] = -1;
	}
    
    // show("Data: ", buf, n);
    cplx* res = fft(buf, n);
    // show("\nFFT : ", res, n);

    // show("\nIFFT : ", ifft(res, n), n);

    ifft(res, n);

    return 0;
}