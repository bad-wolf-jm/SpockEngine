#include <filesystem>
#include <imgui.h>

#include "FileTree.h"

#define BIT(x) (1 << x)

namespace fs = std::filesystem;

namespace WB::Core::UI
{

std::pair<bool, uint32_t> DirectoryTreeViewRecursive(const std::filesystem::path& path, uint32_t* count, int* selection_mask)
{
    ImGuiTreeNodeFlags base_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_SpanFullWidth;

    bool any_node_clicked = false;
    uint32_t node_clicked = 0;

    std::vector<fs::path> l_Files = {};
    std::vector<fs::path> l_Folders = {};

    for (const auto& entry : std::filesystem::directory_iterator(path))
    {
        std::string name = entry.path().string();

        if (!std::filesystem::is_directory(entry.path()))
        {
            l_Files.push_back(entry.path());
        }
        else
        {
            l_Folders.push_back(entry.path());
        }
    }

    for (const auto& entry : l_Folders)
    {
        ImGuiTreeNodeFlags flags = base_flags;
        auto name = entry.filename();
        bool node_open = ImGui::TreeNodeEx(entry.string().c_str(), flags, name.string().c_str());

        if (ImGui::IsItemClicked()) {
            any_node_clicked = true;
        }

        if (node_open)
        {
            auto clickState = DirectoryTreeViewRecursive(entry, count, selection_mask);
            if (!any_node_clicked)
            {
                any_node_clicked = clickState.first;
                node_clicked = clickState.second;
            }

            ImGui::TreePop();
        }
    }

    for (const auto& entry : l_Files)
    {
        ImGuiTreeNodeFlags flags = base_flags | ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
        auto name = entry.filename();
        bool node_open = ImGui::TreeNodeEx(entry.string().c_str(), flags, name.string().c_str());

        if (ImGui::IsItemClicked()) {
            any_node_clicked = true;
        }
    }
    return { any_node_clicked, node_clicked };
}

}