#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

__global__ void AccumulateGradKernel(const float *grad_ptr, float rate, float *tensor_ptr, size_t num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        tensor_ptr[idx] += rate * grad_ptr[idx];
    }
}
void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    size_t num_elements = gradient->NumElements();

    const float *grad_ptr = static_cast<const float *>(gradient->DataPtr());
    float *tensor_ptr = static_cast<float *>(tensor->DataPtr());

    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    AccumulateGradKernel<<<num_blocks, threads_per_block>>>(grad_ptr, rate, tensor_ptr, num_elements);
}


__global__ void AdamAccumulateGradKernel(
    const float* __restrict__ grad_ptr,
    float* __restrict__ param_ptr,
    float* __restrict__ m_ptr,
    float* __restrict__ v_ptr,
    float learning_rate,
    float beta1,
    float beta2,
    float eps,
    float bias_correction_m,
    float bias_correction_v,
    size_t num_elements
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elements) {
        float g = grad_ptr[idx];
        float m = m_ptr[idx];
        float v = v_ptr[idx];

        m = fmaf(beta1, m, (1.0f - beta1) * g);
        v = fmaf(beta2, v, (1.0f - beta2) * g * g);

        m_ptr[idx] = m;
        v_ptr[idx] = v;

        float m_hat = m / bias_correction_m;
        float v_hat = v / bias_correction_v;

        float denom = __fsqrt_rn(v_hat) + eps;
        float inv_denom = __frcp_rn(denom); 
        
        float update = learning_rate * m_hat * inv_denom;

        param_ptr[idx] -= update;
    }
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, 
                        float learning_rate, float beta1, float beta2, float eps, int64_t t) {
    
    size_t num_elements = grad->NumElements();
    
    float bias_correction_m = 1.0f - std::pow(beta1, static_cast<float>(t));
    float bias_correction_v = 1.0f - std::pow(beta2, static_cast<float>(t));

    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    AdamAccumulateGradKernel<<<num_blocks, threads_per_block>>>(
        static_cast<const float *>(grad->DataPtr()), 
        static_cast<float *>(param->DataPtr()), 
        static_cast<float *>(m->DataPtr()), 
        static_cast<float *>(v->DataPtr()), 
        learning_rate, 
        beta1, 
        beta2, 
        eps, 
        bias_correction_m, 
        bias_correction_v, 
        num_elements
    );
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                              \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL
