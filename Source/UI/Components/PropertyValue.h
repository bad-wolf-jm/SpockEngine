#pragma once

#include "Core/Memory.h"

#include "UI/Components/Label.h"
#include "UI/Layouts/BoxLayout.h"


namespace SE::Core
{
    class UIPropertyValue : public UIBoxLayout
    {
      public:
        UIPropertyValue()  = default;
        ~UIPropertyValue() = default;

        UIPropertyValue( string_t aName );
        UIPropertyValue( string_t aName, eBoxLayoutOrientation aOrientation );

        void SetValue( string_t aValue );
        void SetValueFont( FontFamilyFlags aFont );
        void SetNameFont( FontFamilyFlags aFont );

        template <typename... _Ty>
        void SetValue( string_t aFormat, _Ty &&...aArgList )
        {
            string_t lValue = fmt::format( aFormat, std::forward<_Ty>( aArgList )... );
            SetValue( lValue );
        }

      protected:
        Ref<UILabel> mName;
        Ref<UILabel> mValue;

    //   public:
    //     static void *UIPropertyValue_Create();
    //     static void *UIPropertyValue_CreateWithText( void *aText );
    //     static void *UIPropertyValue_CreateWithTextAndOrientation( void *aText, eBoxLayoutOrientation aOrientation );
    //     static void  UIPropertyValue_Destroy( void *aInstance );
    //     static void  UIPropertyValue_SetValue( void *aInstance, void *aText );
    //     static void  UIPropertyValue_SetValueFont( void *aInstance, FontFamilyFlags aFont );
    //     static void  UIPropertyValue_SetNameFont( void *aInstance, FontFamilyFlags aFont );
    };
} // namespace SE::Core