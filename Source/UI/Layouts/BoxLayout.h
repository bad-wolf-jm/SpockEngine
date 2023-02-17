#pragma once

#include "UI/Components/Component.h"

namespace SE::Core
{
    struct BoxLayoutItem
    {
        UIComponent *mItem = nullptr;

        float mFixedSize = 0.0f;
        bool  mExpand    = true;
        bool  mFill      = true;
    };

    class BoxLayout : UIComponent
    {
        BoxLayout()  = default;
        ~BoxLayout() = default;

        BoxLayout &Add( UIComponent *aChild, bool aExpand, bool aFill );
        BoxLayout &Add( UIComponent *aChild, float aFixedSize, bool aExpand, bool aFill );

      protected:
        std::vector<BoxLayoutItem> mChildren;

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };
} // namespace SE::Core