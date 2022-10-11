#pragma once

#include <filesystem>

namespace fs = std::filesystem;

namespace LTSE::Core
{

    fs::path GetResourcePath( fs::path a_RelativePath );

}