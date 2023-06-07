#pragma once

#include "BaseImage.h"
#include "Component.h"
#include "Label.h"
#include "UI/Layouts/BoxLayout.h"
#include "UI/Layouts/StackLayout.h"
#include "UI/Layouts/ZLayout.h"

namespace SE::Core
{
    class UIDropdownButton : public UIComponent
    {
      public:
        UIDropdownButton();
        ~UIDropdownButton() = default;

        void SetText( std::string aValue );
        void SetImage( UIBaseImage *aValue );
        void SetContent( UIComponent *aValue );
        void SetContentSize( math::vec2 aSize );
        void SetTextColor( math::vec4 aColor );

        ImVec2 RequiredSize();

      private:
        bool mActivated = false;

        Ref<UIStackLayout> mImage  = nullptr;
        Ref<UILabel>       mText   = nullptr;
        Ref<UIBoxLayout>   mLayout = nullptr;

        UIComponent *mContent = nullptr;
        ImVec2       mContentSize{};

      private:
        void PushStyles();
        void PopStyles();
        void DrawContent( ImVec2 aPosition, ImVec2 aSize );

    //   public:
    //     static void *UIDropdownButton_Create();
    //     static void  UIDropdownButton_Destroy( void *aInstance );
    //     static void  UIDropdownButton_SetContent( void *aInstance, void *aContent );
    //     static void  UIDropdownButton_SetContentSize( void *aInstance, math::vec2 aSize );
    //     static void  UIDropdownButton_SetImage( void *aInstance, void *aImage );
    //     static void  UIDropdownButton_SetText( void *aInstance, void *aText );
    //     static void  UIDropdownButton_SetTextColor( void *aInstance, math::vec4 aColor );
    };
} // namespace SE::Core