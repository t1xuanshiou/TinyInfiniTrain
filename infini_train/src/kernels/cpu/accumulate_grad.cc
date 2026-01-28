#include <cstddef>
#include <memory>

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    for (int64_t idx = 0; idx < gradient->NumElements(); ++idx) {
        static_cast<float *>(tensor->DataPtr())[idx] += rate * static_cast<const float *>(gradient->DataPtr())[idx];
    }
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    // =================================== 作业 ===================================
    // TODO：实现Adam优化器的梯度累积和参数更新
    // REF:
    // =================================== 作业 ===================================
    
    // for (int64_t idx = 0; idx < param->NumElements(); ++idx){
    //     // 更新一阶矩
    //     static_cast<float *>(m->DataPtr())[idx] = beta1 * static_cast<float *>(m->DataPtr())[idx] + (1 - beta1) * static_cast<const float *>(grad->DataPtr())[idx];
    //     // 更新二阶矩
    //     static_cast<float *>(v->DataPtr())[idx] = beta2 * static_cast<float *>(v->DataPtr())[idx] + (1 - beta2) * static_cast<const float *>(grad->DataPtr())[idx] * static_cast<const float *>(grad->DataPtr())[idx];
        
    //     // 修正一阶矩和二阶矩的偏差
    //     float m_hat = static_cast<float *>(m->DataPtr())[idx] / (1 - std::pow(beta1, t));
    //     float v_hat = static_cast<float *>(v->DataPtr())[idx] / (1 - std::pow(beta2, t));
    //     // 更新参数
    //     static_cast<float *>(param->DataPtr())[idx] -= learning_rate * m_hat / (std::sqrt(v_hat) + eps);
    // }

    // 向量化
    auto m_map = m->EigenVector();
    auto v_map = v->EigenVector();
    auto grad_map = grad->EigenVector();
    auto p_map = param->EigenVector();

    float bc1 = 1.0f - std::pow(beta1, static_cast<float>(t));
    float bc2 = 1.0f - std::pow(beta2, static_cast<float>(t));

    m_map.array() = beta1 * m_map.array() + (1.0f - beta1) * grad_map.array();
    v_map.array() = beta2 * v_map.array() + (1.0f - beta2) * grad_map.array().square();

    auto m_hat = m_map.array() / bc1;
    auto v_hat = v_map.array() / bc2;

    p_map.array() -= learning_rate * m_hat / (v_hat.sqrt() + eps);
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                               \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CPU_ACCUMULATE_GRAD_KERNEL
