#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <omp.h>

typedef double complex cplx;
#define cx_size sizeof(cplx)

cplx *roots;
int N = 0;

void show(const char * s, cplx buf[], unsigned n) {
    printf("%s", s);
    for (int i = 0; i < n; i++)
        if (!cimag(buf[i]))
            printf("%g ", creal(buf[i]));
        else
            printf("(%g, %g) ", creal(buf[i]), cimag(buf[i]));

    printf("\n");
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
	int k;
	#pragma omp parallel for shared(A, a, n) private(k)
    for (k = 0; k < n; k++) {
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
	cplx t;
	cplx w;
	int k;
	int j;
	int s;
	int m;
	cplx w_m;
    for (s = 1; s <= (int)power; s++) {
        m = pow(2, s);
        w_m = cexp(2 * M_PI * -I / m);
		
		#pragma omp parallel for shared(dft, s) private(j, k, t, w)
        for (k = 0; k < n; k += m) {
            w = 1;
            for (j = 0; j < m / 2; j++) {
                t = w * dft[k + j + m / 2];
                cplx u = dft[k + j];
                dft[k + j] = u + t;
                dft[k + j + m / 2] = u - t;
                w = w * w_m;
            }
        }
    }
    return dft;
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
    N = (int) ceil(log2(n));
    // show("Data: ", buf, n);
    cplx* res = fft(buf, n);
    // show("\nFFT : ", res, n);

    return 0;
}