
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <filesystem>
#include <iostream>
#include <numeric>

#include "TestUtils.h"

// #include "Core/CUDA/Texture/TextureData.h"
#include "Core/CUDA/Texture/TextureTypes.h"
#include "Core/Math/Types.h"

#include "Core/CUDA/Texture/Texture2D.h"

namespace fs = std::filesystem;

using namespace numlua;
using namespace numlua::core;
using namespace numlua::cuda;

TEST_CASE( "Vectors", "[CORE_CUDA_TEXTURES]" )
{
    SECTION( "4D vectors" )
    {
    }
}
