#pragma once

#include <sstream>
#include <stdexcept>
#include <string>

namespace qsfi_flashinfer_checks {

class CheckStream {
public:
    CheckStream(const char* file, int line, const char* expr)
    {
        stream_ << file << ':' << line << ": check failed";
        if (expr != nullptr)
            stream_ << " (" << expr << ')';
        stream_ << ": ";
    }

    CheckStream(const CheckStream&) = delete;
    CheckStream& operator=(const CheckStream&) = delete;

    ~CheckStream() noexcept(false)
    {
        throw std::runtime_error(stream_.str());
    }

    template <typename T> CheckStream& operator<<(const T& value)
    {
        stream_ << value;
        return *this;
    }

private:
    std::ostringstream stream_;
};

} // namespace qsfi_flashinfer_checks

#define TVM_FFI_ICHECK(cond)                                                                       \
    if (cond)                                                                                      \
        ;                                                                                          \
    else                                                                                           \
        qsfi_flashinfer_checks::CheckStream(__FILE__, __LINE__, #cond)

#define TVM_FFI_ICHECK_LE(x, y)                                                                    \
    if ((x) <= (y))                                                                                \
        ;                                                                                          \
    else                                                                                           \
        qsfi_flashinfer_checks::CheckStream(__FILE__, __LINE__, #x " <= " #y)

#define TVM_FFI_LOG_AND_THROW(ErrorKind)                                                           \
    qsfi_flashinfer_checks::CheckStream(__FILE__, __LINE__, #ErrorKind)
