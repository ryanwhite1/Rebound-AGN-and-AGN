#include <math.h>

void spline(double x[], double y[], int n, double y2[]){
    // https://www.foo.be/docs-free/Numerical_Recipe_In_C/c3-3.pdf
    int i, k;
    double p, sig;
    double u[n - 1];
    y2[1] = u[1] = 0.;

    for (i = 2; i <= n - 1; i++){
        sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
        p = sig * y2[i - 1] + 2.;
        y2[i] = (sig - 1.) / p;
        u[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
        u[i] = (6. * u[i] / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p;
    }
    y2[n] = 0;
    for (k = n - 1; k >= 1; k--){
        y2[k] = y2[k] * y2[k + 1] + u[k];
    }
}

double splint(double xa[], double ya[], double y2a[], int n, double x){
    int klo, khi, k;
    double h, b, a, y;

    klo = 1; khi = n;
    while (khi - klo > 1){
        k = (khi + klo) >> 1;
        if (xa[k] > x) khi = k;
        else klo = k;
    }
    h = xa[khi] - xa[klo];
    a = (xa[khi] - x) / h;
    b = (x - xa[klo]) / h;
    y = a * ya[klo] + b * ya[khi] + ((a*a*a - a) * y2a[klo] + (b*b*b - b) * y2a[khi]) * (h*h) / 6.;
    return y;
}

double splderiv(double xa[], double ya[], double y2a[], int n, double x){
    int klo, khi, k;
    double h, b, a, dydx;

    klo = 1; khi = n;
    while (khi - klo > 1){
        k = (khi + klo) >> 1;
        if (xa[k] > x) khi = k;
        else klo = k;
    }
    h = xa[khi] - xa[klo];
    a = (xa[khi] - x) / h;
    b = (x - xa[klo]) / h;
    dydx = (ya[khi] - ya[klo])/h - (3.*a*a - 1.)*h*y2a[klo]/6. + (3.*b*b - 1)*h*y2a[khi]/6.;
    return dydx;
}
