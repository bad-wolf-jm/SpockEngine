#pragma once

#include <filesystem>

namespace fs = std::filesystem;

namespace SE::Core
{

    fs::path GetResourcePath( fs::path a_RelativePath );

}