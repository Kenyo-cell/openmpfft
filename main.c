#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <omp.h>
 
double PI;
typedef double complex cplx;
 
void _fft(cplx buf[], cplx out[], int n, int step)
{
    if (step < n) {
        #pragma omp task default(shared)
        _fft(out, buf, n, step * 2);
        _fft(out + step, buf + step, n, step * 2);
		#pragma omp taskwait
        for (int i = 0; i < n; i += 2 * step) {
            cplx t = cexp(-I * PI * i / n) * out[i + step];
            buf[i / 2]     = out[i] + t;
            buf[(i + n)/2] = out[i] - t;
        }
    }
	printf("step %d\n", step);
	for (int i = 0; i < n; i++) {
		printf("%d: (%g, %g); ", i, creal(buf[i]), cimag(buf[i]));
	}
	printf("\n");
}
 
void fft(cplx buf[], int n)
{
	cplx out[n];
	for (int i = 0; i < n; i++) out[i] = buf[i];
	#pragma omp parallel
	#pragma omp single
	_fft(buf, out, n, 1);
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

int main()
{
	PI = atan2(1, 1) * 4;
	cplx buf[] = {5, 2, 4, -1};
    unsigned n = 4;
	show("Data: ", buf, n);
	fft(buf, n);
	show("\nFFT : ", buf, n);
 
	return 0;
}