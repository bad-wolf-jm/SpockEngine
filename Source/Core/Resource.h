#pragma once

#include "Core/String.h"

namespace fs = std::filesystem;

namespace SE::Core
{

    path_t GetResourcePath( path_t a_RelativePath );

}