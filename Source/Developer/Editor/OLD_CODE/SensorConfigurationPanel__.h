#pragma once
#include <string>

// #include "SidePanel.h"

#include "Core/EntityRegistry/Registry.h"
#include "Core/Memory.h"

#include "SensorController.h"

#include "SensorAssetProperties.h"
#include "SensorComponentEditor.h"
#include "TilePropertyEditor.h"
#include "TileLayoutEditor.h"
#include "PhotodetectorCellEditor.h"


namespace LTSE::Editor
{
    class SensorConfigurationPanel : public SidePanel
    {

      public:
        enum PropertyPanelID
        {
            NONE,
            TILE_PROPERTY_EDITOR,
            SENSOR_COMPONENT_EDITOR,
            SENSOR_ASSET_EDITOR,
            TILE_LAYOUT_EDITOR
        };

        Ref<SensorDeviceBase> SensorModel = nullptr;
        Entity TileToEdit{};
        Entity TileInFocus{};
        TilePropertyEditor ElementEditor{};
        TileLayoutProperties LayoutEditor{};
        SensorComponentEditor ComponentEditor{};
        SensorAssetProperties AssetProperties{};
        PropertyPanelID RequestPropertyEditor = PropertyPanelID::NONE;

      public:
        SensorConfigurationPanel()  = default;
        ~SensorConfigurationPanel() = default;

        void Display( int32_t width, int32_t height );
    };

} // namespace LTSE::Editor