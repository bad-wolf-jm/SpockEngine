#include "PropertyValue.h"

#include "DotNet/Runtime.h"
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

    void UIPropertyValue::SetValue( string_t aValue )
    {
        mValue->SetText( aValue );
    }
    void UIPropertyValue::SetValueFont( FontFamilyFlags aFont )
    {
        mValue->mFont = aFont;
    }
    void UIPropertyValue::SetNameFont( FontFamilyFlags aFont )
    {
        mName->mFont = aFont;
    }

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

    void *UIPropertyValue::UIPropertyValue_CreateWithTextAndOrientation( void *aText, eBoxLayoutOrientation aOrientation )
    {
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );
        auto lNewLabel = new UIPropertyValue( lString, aOrientation );

        return static_cast<void *>( lNewLabel );
    }

    void UIPropertyValue::UIPropertyValue_Destroy( void *aInstance )
    {
        delete static_cast<UIPropertyValue *>( aInstance );
    }

    void UIPropertyValue::UIPropertyValue_SetValue( void *aInstance, void *aText )
    {
        auto lInstance = static_cast<UIPropertyValue *>( aInstance );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

        lInstance->SetValue( lString );
    }

    void UIPropertyValue::UIPropertyValue_SetValueFont( void *aInstance, FontFamilyFlags aFont )
    {
        auto lInstance = static_cast<UIPropertyValue *>( aInstance );

        lInstance->SetValueFont( aFont );
    }

    void UIPropertyValue::UIPropertyValue_SetNameFont( void *aInstance, FontFamilyFlags aFont )
    {
        auto lInstance = static_cast<UIPropertyValue *>( aInstance );

        lInstance->SetNameFont( aFont );
    }

} // namespace SE::Core