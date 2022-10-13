#pragma once

#include "PopupWindow.h"

#include "Developer/UI/CanvasView.h"

#include "Core/EntityRegistry/Registry.h"

#include "LidarSensorModel/SensorDeviceBase.h"

using namespace LTSE::Core;
using namespace LTSE::SensorModel;

namespace LTSE::Editor
{

    class PhotodetectorCellEditor : public PopupWindow
    {
      public:
        Entity PhotodetectorToEdit{};
        Ref<SensorDeviceBase> SensorModel = nullptr;
        UI::CanvasView Canvas{};
      public:
        PhotodetectorCellEditor() = default;
        PhotodetectorCellEditor( std::string a_Title, math::vec2 a_Size );
        ~PhotodetectorCellEditor() = default;
        void WindowContent();
    };

} // namespace LTSE::Editor