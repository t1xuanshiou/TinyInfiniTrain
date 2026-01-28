#include "example/common/tiny_shakespeare_dataset.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace {
using DataType = infini_train::DataType;
using TinyShakespeareType = TinyShakespeareDataset::TinyShakespeareType;
using TinyShakespeareFile = TinyShakespeareDataset::TinyShakespeareFile;

const std::unordered_map<int, TinyShakespeareType> kTypeMap = {
    {20240520, TinyShakespeareType::kUINT16}, // GPT-2
    {20240801, TinyShakespeareType::kUINT32}, // LLaMA 3
};

const std::unordered_map<TinyShakespeareType, size_t> kTypeToSize = {
    {TinyShakespeareType::kUINT16, 2},
    {TinyShakespeareType::kUINT32, 4},
};

const std::unordered_map<TinyShakespeareType, DataType> kTypeToDataType = {
    {TinyShakespeareType::kUINT16, DataType::kUINT16},
    {TinyShakespeareType::kUINT32, DataType::kINT32},
};

std::vector<uint8_t> ReadSeveralBytesFromIfstream(size_t num_bytes, std::ifstream *ifs) {
    std::vector<uint8_t> result(num_bytes);
    ifs->read(reinterpret_cast<char *>(result.data()), num_bytes);
    return result;
}

template <typename T> T BytesToType(const std::vector<uint8_t> &bytes, size_t offset) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable.");
    T value;
    std::memcpy(&value, &bytes[offset], sizeof(T));
    return value;
}

TinyShakespeareFile ReadTinyShakespeareFile(const std::string &path, size_t sequence_length) {
    /* =================================== 作业 ===================================
       TODO：实现二进制数据集文件解析
       文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | DATA (tokens)                        |
    | magic(4B) | version(4B) | num_toks(4B) | reserved(1012B) | token数据           |
    ----------------------------------------------------------------------------------
       =================================== 作业 =================================== */
    //类型定义:
    //struct TinyShakespeareFile {
    //     TinyShakespeareType type = TinyShakespeareType::kINVALID;
    //     std::vector<int64_t> dims;
    //     infini_train::Tensor tensor;
    // };
    if (!std::filesystem::exists(path)) {
        LOG(FATAL) << "File not found: " << path;
    }

    TinyShakespeareFile text_file;
    std::ifstream ifs(path, std::ios::binary);
    const auto header = ReadSeveralBytesFromIfstream(1024, &ifs);
    const int magic = BytesToType<int32_t>(header, 0);
    const int num_tokens = BytesToType<int32_t>(header, 8);
    text_file.type = kTypeMap.at(magic);

    const int num_sequences = num_tokens / sequence_length;
    text_file.dims = std::vector<int64_t>{static_cast<int64_t>(num_sequences), static_cast<int64_t>(sequence_length)};
    const int data_size_in_bytes = static_cast<int>(kTypeToSize.at(text_file.type)) * static_cast<int>(sequence_length) * static_cast<int>(num_sequences);
    text_file.tensor = infini_train::Tensor(text_file.dims, DataType::kINT64);
    int64_t *dst = static_cast<int64_t *>(text_file.tensor.DataPtr());
    std::vector<uint8_t> data = ReadSeveralBytesFromIfstream(data_size_in_bytes, &ifs);
    const auto &element_size = kTypeToSize.at(text_file.type);
    int total_token_count = num_sequences * sequence_length;
    for (int i = 0; i < total_token_count; ++i) {
        int64_t value = 0;
        if (text_file.type == TinyShakespeareType::kUINT16) {
            uint16_t v = BytesToType<uint16_t>(data, i * element_size);
            value = static_cast<int64_t>(v);
        } else if (text_file.type == TinyShakespeareType::kUINT32) {
            uint32_t v = BytesToType<uint32_t>(data, i * element_size);
            value = static_cast<int64_t>(v);
        }
        dst[i] = value;
    }
    return text_file;
}
} // namespace

// TODO：初始化
TinyShakespeareDataset::TinyShakespeareDataset(const std::string &filepath, size_t sequence_length): text_file_(ReadTinyShakespeareFile(filepath, sequence_length)),
    sequence_length_(sequence_length),
    sequence_size_in_bytes_(sequence_length * sizeof(int64_t)),
    num_samples_(text_file_.dims[0] - 1) {}

std::pair<std::shared_ptr<infini_train::Tensor>, std::shared_ptr<infini_train::Tensor>>
TinyShakespeareDataset::operator[](size_t idx) const {
    CHECK_LT(idx, text_file_.dims[0] - 1);
    std::vector<int64_t> dims = std::vector<int64_t>(text_file_.dims.begin() + 1, text_file_.dims.end());
    // x: (seq_len), y: (seq_len) -> stack -> (bs, seq_len) (bs, seq_len)
    return {std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_, dims),
            std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_ + sizeof(int64_t),
                                                   dims)};
}

size_t TinyShakespeareDataset::Size() const { return num_samples_; }
