#pragma once
#include <string>

#include "Core/EntityRegistry/Registry.h"
#include "Core/Memory.h"


#include "Developer/GraphicContext/GraphicContext.h"
#include "Developer/Scene/Scene.h"

#include "Developer/Platform/EngineLoop.h"

#include "Serialize/SensorAsset.h"

#include "PropertiesPanel.h"
#include "Sensorcontroller.h"

using namespace LTSE::Core;
using namespace LTSE::Core::UI;
using namespace LTSE::Core::EntityComponentSystem;
using namespace LTSE::SensorModel;

namespace LTSE::Editor
{
    class SensorAssetProperties : public PropertiesPanel
    {
      public:
        Entity ComponentToEdit{};
        Ref<SensorDeviceBase> SensorModel = nullptr;

      public:
        SensorAssetProperties();
        ~SensorAssetProperties() = default;

        void Display( int32_t width, int32_t height );
    };

} // namespace LTSE::Editor