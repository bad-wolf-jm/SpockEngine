#include "PropertyValue.h"

namespace SE::OtdrEditor
{
    UIPropertyValue::UIPropertyValue( std::string aName )
        : UIBoxLayout( eBoxLayoutOrientation::HORIZONTAL )
    {
        mName  = New<UILabel>( aName );
        mValue = New<UILabel>( "N/A" );

        Add( mName.get(), true, false, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        Add( mValue.get(), true, false, eHorizontalAlignment::RIGHT, eVerticalAlignment::CENTER );
    }

    void UIPropertyValue::SetValue( std::string aValue ) { mValue->SetText( aValue ); }
} // namespace SE::OtdrEditor