#pragma once

#include "Core/Memory.h"

#include "UI/Layouts/BoxLayout.h"
#include "UI/Components/Label.h"

namespace SE::OtdrEditor
{
    using namespace SE::Core;

    class PropertyValue : public UIBoxLayout
    {
        PropertyValue() = default;
        ~PropertyValue() = default;

        PropertyValue( std::string aName );

        void SetValue(std::string aValue);


      protected:
        Ref<UILabel> mName;
        Ref<UILabel> mValue;
    };

    class MeasurementOverview : public UIBoxLayout
    {
    };
} // namespace SE::OtdrEditor