VERBATIM
#include <math.h>
#include "nrniv_mf.h"

static double interp4D(double *table, double *A_vals, double *Q_vals, double *A1_vals, double *B1_vals, 
                       int A_size, int Q_size, int A1_size, int B1_size, double A, double Q, double A1, double B1) {
    int i, j, k, l;
    i = 0; j = 0; k = 0; l = 0; 
    char ii = 'n';
    char ji = 'n';
    char ki = 'n';
    char li = 'n';
    for (i = 0; i < A_size - 1; i++) {
	if (A_vals[i] <= A && A_vals[i + 1] > A) {
	    ii = 'y';
            for (j = 0; j < Q_size - 1; j++) {
                if (Q_vals[j] <= Q && Q_vals[j + 1] > Q) {
		    ji = 'y';
                    for (k = 0; k < A1_size - 1; k++) {
                        if (A1_vals[k] <= A1 && A1_vals[k + 1] > A1) {
			    ki = 'y';
                            for (l = 0; l < B1_size - 1; l++) {
                                if (B1_vals[l] <= B1 && B1_vals[l + 1] > B1) {
				    li = 'y';
                                    double tA = (A - A_vals[i]) / (A_vals[i + 1] - A_vals[i]);
                                    double tQ = (Q - Q_vals[j]) / (Q_vals[j + 1] - Q_vals[j]);
                                    double tA1 = (A1 - A1_vals[k]) / (A1_vals[k + 1] - A1_vals[k]);
                                    double tB1 = (B1 - B1_vals[l]) / (B1_vals[l + 1] - B1_vals[l]);

                                    double v0000 = table[i * Q_size * A1_size * B1_size + j * A1_size * B1_size + k * B1_size + l];
                                    double v0001 = table[i * Q_size * A1_size * B1_size + j * A1_size * B1_size + k * B1_size + l + 1];
                                    double v0010 = table[i * Q_size * A1_size * B1_size + j * A1_size * B1_size + (k + 1) * B1_size + l];
                                    double v0011 = table[i * Q_size * A1_size * B1_size + j * A1_size * B1_size + (k + 1) * B1_size + l + 1];
                                    double v0100 = table[i * Q_size * A1_size * B1_size + (j + 1) * A1_size * B1_size + k * B1_size + l];
                                    double v0101 = table[i * Q_size * A1_size * B1_size + (j + 1) * A1_size * B1_size + k * B1_size + l + 1];
                                    double v0110 = table[i * Q_size * A1_size * B1_size + (j + 1) * A1_size * B1_size + (k + 1) * B1_size + l];
                                    double v0111 = table[i * Q_size * A1_size * B1_size + (j + 1) * A1_size * B1_size + (k + 1) * B1_size + l + 1];

                                    double v1000 = table[(i + 1) * Q_size * A1_size * B1_size + j * A1_size * B1_size + k * B1_size + l];
                                    double v1001 = table[(i + 1) * Q_size * A1_size * B1_size + j * A1_size * B1_size + k * B1_size + l + 1];
                                    double v1010 = table[(i + 1) * Q_size * A1_size * B1_size + j * A1_size * B1_size + (k + 1) * B1_size + l];
                                    double v1011 = table[(i + 1) * Q_size * A1_size * B1_size + j * A1_size * B1_size + (k + 1) * B1_size + l + 1];
                                    double v1100 = table[(i + 1) * Q_size * A1_size * B1_size + (j + 1) * A1_size * B1_size + k * B1_size + l];
                                    double v1101 = table[(i + 1) * Q_size * A1_size * B1_size + (j + 1) * A1_size * B1_size + k * B1_size + l + 1];
                                    double v1110 = table[(i + 1) * Q_size * A1_size * B1_size + (j + 1) * A1_size * B1_size + (k + 1) * B1_size + l];
                                    double v1111 = table[(i + 1) * Q_size * A1_size * B1_size + (j + 1) * A1_size * B1_size + (k + 1) * B1_size + l + 1];			
    				
				    //extern NrnThread* nrn_threads;
                                    return (1 - tA) * (1 - tQ) * (1 - tA1) * (1 - tB1) * v0000 +
                                           tA * (1 - tQ) * (1 - tA1) * (1 - tB1) * v1000 +
                                           (1 - tA) * tQ * (1 - tA1) * (1 - tB1) * v0100 +
                                           tA * tQ * (1 - tA1) * (1 - tB1) * v1100 +
                                           (1 - tA) * (1 - tQ) * tA1 * (1 - tB1) * v0010 +
                                           tA * (1 - tQ) * tA1 * (1 - tB1) * v1010 +
                                           (1 - tA) * tQ * tA1 * (1 - tB1) * v0110 +
                                           tA * tQ * tA1 * (1 - tB1) * v1110 +
                                           (1 - tA) * (1 - tQ) * (1 - tA1) * tB1 * v0001 +
                                           tA * (1 - tQ) * (1 - tA1) * tB1 * v1001 +
                                           (1 - tA) * tQ * (1 - tA1) * tB1 * v0101 +
                                           tA * tQ * (1 - tA1) * tB1 * v1101 +
                                           (1 - tA) * (1 - tQ) * tA1 * tB1 * v0011 +
                                           tA * (1 - tQ) * tA1 * tB1 * v1011 +
                                           (1 - tA) * tQ * tA1 * tB1 * v0111 +
                                           tA * tQ * tA1 * tB1 * v1111;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    //printf("As: (%c), Qs: (%c), A1s: (%c), B1s: (%c)\n", ii, ji, ki, li);
    //printf("ranges: A: (%g, %g), Q: (%g, %g), A1: (%g, %g), B1: (%g, %g)", A_vals[0], A_vals[A_size-1], Q_vals[0], Q_vals[Q_size-1], A1_vals[0], A1_vals[A1_size-1], B1_vals[0], B1_vals[B1_size-1]);
    //printf("A: (%g, %i), Q: (%g, %i), A1: (%g, %i), B1: (%g, %i)\n", A, i, Q, j, A1, k, B1, l);
    //printf("interpolation out of range\n");
    return table[A_size * Q_size * A1_size * B1_size - 1];
}
ENDVERBATIM