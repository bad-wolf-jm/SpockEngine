#pragma once
#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

#include "Core/Entity/Collection.h"
#include "Core/CUDA/Array/MultiTensor.h"

namespace numlua::core
{
    void open_tensor_library( sol::table &scriptingState );
}; // namespace SE::Core