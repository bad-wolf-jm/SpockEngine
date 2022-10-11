#pragma once
#include <string>

#include "Core/Memory.h"
#include "Core/EntityRegistry/Registry.h"

#include "Developer/GraphicContext/GraphicContext.h"
#include "Developer/Scene/Scene.h"

#include "Developer/Platform/EngineLoop.h"

#include "Serialize/SensorAsset.h"

#include "TileFlashEditor.h"
#include "TileLayoutEditor.h"
#include "PropertiesPanel.h"
#include "SensorController.h"

using namespace LTSE::Core;
using namespace LTSE::Core::UI;
using namespace LTSE::Core::EntityComponentSystem;
using namespace LTSE::SensorModel;

namespace LTSE::Editor
{
    class TilePropertyEditor : public PropertiesPanel
    {
      public:
        Entity TileToEdit{};
        Entity FlashToEdit{};
        bool EditFlashes = false;
        Ref<SensorDeviceBase> SensorModel = nullptr;

      public:
        TilePropertyEditor();
        ~TilePropertyEditor() = default;

        void Display( int32_t width, int32_t height );
        void DisplayTileFieldOfView( int32_t width );

      private:
        FlashAttenuationBindingPopup FlashEditor;
    };

   class TileLayoutProperties : public PropertiesPanel
    {
      public:
        Entity LayoutToEdit{};
        Ref<SensorDeviceBase> SensorModel = nullptr;

      public:
        TileLayoutProperties();
        ~TileLayoutProperties() = default;

        void Display( int32_t width, int32_t height );

      private:
        TileLayoutEditor LayoutEditor;
    };

} // namespace LTSE::Editor