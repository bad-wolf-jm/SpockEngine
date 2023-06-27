#include "PropertyValue.h"

namespace SE::Core
{
    UIPropertyValue::UIPropertyValue( string_t aName )
        : UIPropertyValue( aName, eBoxLayoutOrientation::HORIZONTAL )
    {
    }

    UIPropertyValue::UIPropertyValue( string_t aName, eBoxLayoutOrientation aOrientation )
        : UIBoxLayout( aOrientation )
    {
        mName  = New<UILabel>( aName );
        mValue = New<UILabel>( "N/A" );

        if( aOrientation == eBoxLayoutOrientation::HORIZONTAL )
        {
            Add( mName.get(), true, false, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
            Add( mValue.get(), true, false, eHorizontalAlignment::RIGHT, eVerticalAlignment::CENTER );
        }
        else
        {
            Add( mName.get(), true, false, eHorizontalAlignment::CENTER, eVerticalAlignment::CENTER );
            Add( mValue.get(), true, false, eHorizontalAlignment::CENTER, eVerticalAlignment::CENTER );
        }
    }

    void UIPropertyValue::SetValue( string_t aValue ) { mValue->SetText( aValue ); }
    void UIPropertyValue::SetValueFont( FontFamilyFlags aFont ) { mValue->mFont = aFont; }
    void UIPropertyValue::SetNameFont( FontFamilyFlags aFont ) { mName->mFont = aFont; }
} // namespace SE::Core