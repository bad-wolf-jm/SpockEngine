#pragma once

#include "Core/Memory.h"

#include "UI/Components/Label.h"
#include "UI/Layouts/BoxLayout.h"

#include "Mono/MonoScriptInstance.h"

namespace SE::OtdrEditor
{
    using namespace SE::Core;

    class UIPropertyValue : public UIBoxLayout
    {
      public:
        UIPropertyValue()  = default;
        ~UIPropertyValue() = default;

        UIPropertyValue( std::string aName );

        void SetValue( std::string aValue );

        template <typename... _Ty>
        void SetValue( std::string aFormat, _Ty &&...aArgList )
        {
            std::string lValue = fmt::format( aFormat, std::forward<_Ty>( aArgList )... );
            SetValue( lValue );
        }

      protected:
        Ref<UILabel> mName;
        Ref<UILabel> mValue;

      public:
        static void *UIPropertyValue_Create();
        static void *UIPropertyValue_CreateWithText( void *aText );
        static void  UIPropertyValue_Destroy( void *aInstance );
        static void  UIPropertyValue_SetValue( void *aInstance, void *aText );


    };
} // namespace SE::OtdrEditor