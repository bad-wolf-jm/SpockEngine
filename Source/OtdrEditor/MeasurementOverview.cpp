#include "MeasurementOverview.h"

namespace SE::OtdrEditor
{
    PropertyValue::PropertyValue( std::string aName )
        : UIBoxLayout( eBoxLayoutOrientation::HORIZONTAL )
    {
        mName  = New<UILabel>( aName );
        mValue = New<UILabel>();

        Add( mName.get(), true, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        Add( mValue.get(), true, true, eHorizontalAlignment::RIGHT, eVerticalAlignment::CENTER );
    }

    void PropertyValue::SetValue( std::string aValue ) { mValue->SetText( aValue ); }

} // namespace SE::OtdrEditor