#pragma once

#include "PopupWindow.h"

#include "Developer/UI/CanvasView.h"

#include "Core/EntityRegistry/Registry.h"

#include "LidarSensorModel/SensorDeviceBase.h"

using namespace LTSE::Core;
using namespace LTSE::SensorModel;

namespace LTSE::Editor
{

    class TileLayoutEditor : public PopupWindow
    {
      public:
        Entity LayoutToEdit{};
        Ref<SensorDeviceBase> SensorModel = nullptr;
        UI::CanvasView Canvas{};
      public:
        TileLayoutEditor() = default;
        TileLayoutEditor( std::string a_Title, math::vec2 a_Size );
        ~TileLayoutEditor() = default;
        void WindowContent();
    };

} // namespace LTSE::Editor