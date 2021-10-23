#define SQRT_TWOPI 2.506628274631f

__kernel void kde_fixedKDE(const unsigned int M /*length of data*/, const float h /*bandwidth*/,
                        __global float* data, __global float* weights,
                        __global float* x, __global float* density)
{
    // this is the ith position in X
    unsigned int i = get_global_id(0);

    float w_sum = 0.0f;
    float xi = x[i];
    float den = 0.0f;
    float dm, tmp;
    for(unsigned int m = 0; m < M; m++){
        w_sum += weights[m];
        dm = data[m];
        tmp = exp(-(xi - dm)*(xi - dm)/(2.0f*h*h)) / SQRT_TWOPI;
        den += tmp * weights[m] / h;
        //printf("dm=%f tmp=%f den=%f\n", dm, tmp, den);
    }
    density[i] = den / w_sum;
    //printf("%d %f %f %f %f\n", i, h, den, w_sum, density[i]);
}

__kernel void kde_adaptiveKDE(const unsigned int M /*length of data*/,
                              __global float* data, __global float* hs /*bandwidths*/,
                              __global float* weights /*data weights*/,
                              __global float* x, __global float* density)
{
    // this is the ith position in X
    unsigned int i = get_global_id(0);

    float w_sum = 0.0f;
    float xi = x[i];
    float den = 0.0f;
    float dm, tmp, h;
    for(unsigned int m = 0; m < M; m++){
        w_sum += weights[m];
        dm = data[m];
        h = hs[m];
        tmp = exp(-(xi - dm)*(xi - dm)/(2.0f*h*h)) / SQRT_TWOPI;
        den += tmp * weights[m] / h;
    }
    density[i] = den / w_sum;
}

__kernel void kde_densityAtDataPoints(const unsigned int M /*length of data*/, const float h /*bandwidth*/,
                              __global float* data, __global float* weights /*data weights*/,
                              __global float* density_inc, __global float* density_exc)
{
    // this is the ith position in X
    unsigned int i = get_global_id(0);

    float w_sum = 0.0f;
    float xi = data[i];
    float den_inc = 0.0f;
    float den_exc = 0.0f;
    float dm, tmp;
    for(unsigned int m = 0; m < M; m++){
        w_sum += weights[m];
        dm = data[m];
        tmp = exp(-(xi - dm)*(xi - dm)/(2.0f*h*h)) / SQRT_TWOPI;
        den_inc += tmp * weights[m] / h;
    }
    density_inc[i] = den_inc / w_sum;

    den_exc = den_inc - weights[i] * 1.0f / SQRT_TWOPI / h;
    density_exc[i] = den_exc / (w_sum - weights[i]);
}
