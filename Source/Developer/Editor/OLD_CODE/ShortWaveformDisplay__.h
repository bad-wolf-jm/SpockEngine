#pragma once
#include <string>

#include "Core/Memory.h"
#include "Developer/GraphicContext/UI/UIContext.h"

using namespace LTSE::Core;

namespace LTSE::Editor
{
    void DisplayShortWaveforms( UIContext &aUiContext, std::vector<uint8_t> &aTileData );

} // namespace LTSE::Editor