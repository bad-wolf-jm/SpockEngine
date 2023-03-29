#include "PropertyValue.h"

#include "DotNet/Runtime.h"
namespace SE::OtdrEditor
{
    UIPropertyValue::UIPropertyValue( std::string aName )
        : UIPropertyValue( aName, eBoxLayoutOrientation::HORIZONTAL )
    {
    }

    UIPropertyValue::UIPropertyValue( std::string aName, eBoxLayoutOrientation aOrientation )
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

    void UIPropertyValue::SetValue( std::string aValue ) { mValue->SetText( aValue ); }

    void *UIPropertyValue::UIPropertyValue_Create()
    {
        auto lNewLabel = new UIPropertyValue();

        return static_cast<void *>( lNewLabel );
    }

    void *UIPropertyValue::UIPropertyValue_CreateWithText( void *aText )
    {
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );
        auto lNewLabel = new UIPropertyValue( lString );

        return static_cast<void *>( lNewLabel );
    }

    void UIPropertyValue::UIPropertyValue_Destroy( void *aInstance ) { delete static_cast<UIPropertyValue *>( aInstance ); }

    void UIPropertyValue::UIPropertyValue_SetValue( void *aInstance, void *aText )
    {
        auto lInstance = static_cast<UIPropertyValue *>( aInstance );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

        lInstance->SetValue( lString );
    }
} // namespace SE::OtdrEditor