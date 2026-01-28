#include <cstdint>
#include <fcntl.h>
#include <memory>
#include <numeric>
#include <tuple>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {

using ConstEigenMatrixMap = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
using EigenMatrixMap = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {

    
    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();

    CHECK_GE(input_dims.size(), 2);
    CHECK_GE(other_dims.size(), 2);
    CHECK_EQ(input_dims.size(), other_dims.size());

    const int64_t m = input_dims[input_dims.size() - 2];
    const int64_t k = input_dims[input_dims.size() - 1];
    const int64_t k_other = other_dims[other_dims.size() - 2];
    const int64_t n = other_dims[other_dims.size() - 1];

    CHECK_EQ(k, k_other) << "Inner dimensions mismatch";

    const int64_t batch_size = std::accumulate(input_dims.begin(), input_dims.end() - 2, 1, std::multiplies<int64_t>());
    
    std::vector<int64_t> output_dims = input_dims;
    output_dims[output_dims.size() - 1] = n;
    auto output = std::make_shared<Tensor>(output_dims, input->Dtype());

    const float* input_ptr = static_cast<const float*>(input->DataPtr());
    const float* other_ptr = static_cast<const float*>(other->DataPtr());
    float* output_ptr = static_cast<float*>(output->DataPtr());

    for (int64_t b = 0; b < batch_size; ++b) {
        auto input_offset = b * m * k;
        auto other_offset = b * k * n;
        auto output_offset = b * m * n;

        ConstEigenMatrixMap input_mat(input_ptr + input_offset, m, k);
        ConstEigenMatrixMap other_mat(other_ptr + other_offset, k, n);
        EigenMatrixMap output_mat(output_ptr + output_offset, m, n);

        output_mat.noalias() = input_mat * other_mat;
    }

    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
               const std::shared_ptr<Tensor> &grad_output) {


    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    const auto &grad_dims = grad_output->Dims();

    const int64_t m = input_dims[input_dims.size() - 2];
    const int64_t k = input_dims[input_dims.size() - 1];
    const int64_t n = other_dims[other_dims.size() - 1];
    const int64_t batch_size = std::accumulate(input_dims.begin(), input_dims.end() - 2, 1, std::multiplies<int64_t>());

    auto grad_input = std::make_shared<Tensor>(input_dims, input->Dtype());
    auto grad_other = std::make_shared<Tensor>(other_dims, other->Dtype());
    
    const float* input_ptr = static_cast<const float*>(input->DataPtr());
    const float* other_ptr = static_cast<const float*>(other->DataPtr());
    const float* grad_output_ptr = static_cast<const float*>(grad_output->DataPtr());
    
    float* grad_input_ptr = static_cast<float*>(grad_input->DataPtr());
    float* grad_other_ptr = static_cast<float*>(grad_other->DataPtr());

    for (int64_t b = 0; b < batch_size; ++b) {
        long in_off = b * m * k;
        long ot_off = b * k * n;
        long go_off = b * m * n;

        ConstEigenMatrixMap input_mat(input_ptr + in_off, m, k);
        ConstEigenMatrixMap other_mat(other_ptr + ot_off, k, n);
        ConstEigenMatrixMap grad_out_mat(grad_output_ptr + go_off, m, n);
        
        EigenMatrixMap grad_in_mat(grad_input_ptr + in_off, m, k);
        EigenMatrixMap grad_ot_mat(grad_other_ptr + ot_off, k, n);

        grad_in_mat.noalias() = grad_out_mat * other_mat.transpose();
        
        grad_ot_mat.noalias() = input_mat.transpose() * grad_out_mat;
    }

    return {grad_input, grad_other};
}





std::shared_ptr<Tensor> LinearForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                      bool transpose, const std::shared_ptr<Tensor> &bias) {
    /*
    transpose:  output = input * weight^T + bias
    output[*, out_features] = input[*, in_features] * weight[out_features, in_features]^T + bias[out_features]

    !transpose: output = input * weight + bias
    output[*, out_features] = input[*, in_features] * weight[in_features, out_features] + bias[out_features]
    */

    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    const int out_features = weight_dims[transpose ? 0 : 1];

    if (bias) {
        const auto &bias_dims = bias->Dims();
        CHECK_EQ(bias_dims.size(), 1);
        CHECK_EQ(bias_dims[0], out_features);
    }

    auto output_dims = input_dims;
    *output_dims.rbegin() = out_features;
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32);

    if (transpose) {
        output->EigenMatrix() = input->EigenMatrix() * weight->EigenMatrix().transpose();
    } else {
        output->EigenMatrix() = input->EigenMatrix() * weight->EigenMatrix();
    }

    if (bias) {
        output->EigenMatrix().rowwise() += bias->EigenVector();
    }

    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LinearBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight, bool transpose,
               int64_t out_features, const std::shared_ptr<Tensor> &grad_output, const bool bias) {
    /*
    transpose: grad_input = grad_output * weight
    grad_input[*, in_features] = grad_output[*, out_features] * weight[out_features, in_features]
    grad_weight[out_features, in_features] = grad_output[*, out_features]^T * input[*, in_features]
    grad_bias[out_features] = grad_output[*, out_features].sum(axis=0)

    !transpose: grad_input = grad_output * weight^T
    grad_input[*, in_features] = grad_output[_, out_features] * weight[in_features, out_features]^T
    grad_weight[in_features, out_features] = input[*, in_features]^T * grad_output[*, out_features]
    grad_bias[out_features] = grad_output[*, out_features].sum(axis=0)
    */

    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    CHECK_EQ(out_features, weight_dims[transpose ? 0 : 1]);

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
    auto grad_weight = std::make_shared<Tensor>(weight_dims, DataType::kFLOAT32);
    std::shared_ptr<Tensor> grad_bias = nullptr;
    if (bias) {
        grad_bias = std::make_shared<Tensor>(std::vector<int64_t>{out_features}, DataType::kFLOAT32);
    }

    if (transpose) {
        grad_input->EigenMatrix() = grad_output->EigenMatrix() * weight->EigenMatrix();
        grad_weight->EigenMatrix() = grad_output->EigenMatrix().transpose() * input->EigenMatrix();
    } else {
        grad_input->EigenMatrix() = grad_output->EigenMatrix() * weight->EigenMatrix().transpose();
        grad_weight->EigenMatrix() = input->EigenMatrix().transpose() * grad_output->EigenMatrix();
    }
    if (bias) {
        grad_bias->EigenVector() = grad_output->EigenMatrix().colwise().sum();
    }

    return {grad_input, grad_weight, grad_bias};
}
} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_LINEAR_KERNEL(kernel_name)                                                                        \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_LINEAR_KERNEL(MatmulForward)
REGISTER_CPU_LINEAR_KERNEL(MatmulBackward)
REGISTER_CPU_LINEAR_KERNEL(LinearForward)
REGISTER_CPU_LINEAR_KERNEL(LinearBackward)

#undef REGISTER_CPU_LINEAR_KERNEL
