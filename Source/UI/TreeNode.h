#pragma once

#include "imgui.h"

namespace SE::Core::UI
{
    bool TreeNodeEx( const char *label, ImGuiTreeNodeFlags flags );
    void TreePop();
}