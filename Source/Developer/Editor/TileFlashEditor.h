#pragma once

#include "PopupWindow.h"

#include "Core/EntityRegistry/Registry.h"

#include "LidarSensorModel/SensorDeviceBase.h"

using namespace LTSE::Core;
using namespace LTSE::SensorModel;

namespace LTSE::Editor
{

    class FlashAttenuationBindingPopup : public PopupWindow
    {
      public:
        Entity TileToEdit{};
        Entity DiffusionPatternAsset{};
        Ref<SensorDeviceBase> SensorModel = nullptr;

      public:
        FlashAttenuationBindingPopup() = default;
        FlashAttenuationBindingPopup( std::string a_Title, math::vec2 a_Size );
        ~FlashAttenuationBindingPopup() = default;
        void WindowContent();
    };

} // namespace LTSE::Editor