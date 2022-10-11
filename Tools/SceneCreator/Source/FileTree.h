#pragma once

namespace LTSE::Core::UI
{

std::pair<bool, uint32_t> DirectoryTreeViewRecursive(const std::filesystem::path& path, uint32_t* count, int* selection_mask);

}
