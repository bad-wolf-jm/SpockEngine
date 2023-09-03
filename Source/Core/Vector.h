#pragma once

#include <array>
#include <vector>

namespace SE::Core
{
    template <typename _Ty>
    using vector_t = std::vector<_Ty>;

    template <typename _Ty, std::size_t _Nm>
    using array_t = std::array<_Ty, _Nm>;
} // namespace SE::Core