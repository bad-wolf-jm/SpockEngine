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

        void SetText( string_t aValue );
        void SetImage( UIBaseImage *aValue );
        void SetContent( UIComponent *aValue );
        void SetContentSize( math::vec2 aSize );
        void SetTextColor( math::vec4 aColor );

        ImVec2 RequiredSize();

      private:
        bool mActivated = false;

        ref_t<UIStackLayout> mImage  = nullptr;
        ref_t<UILabel>       mText   = nullptr;
        ref_t<UIBoxLayout>   mLayout = nullptr;

        UIComponent *mContent = nullptr;
        ImVec2       mContentSize{};

      private:
        void PushStyles();
        void PopStyles();
        void DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };
} // namespace SE::Core