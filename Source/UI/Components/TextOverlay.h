#pragma once

#include "Component.h"

#include <list>

namespace SE::Core
{
    struct sTextLine
    {
        uint32_t    mRepetitions = 0;
        std::string mLine        = "";
        bool        mIsPartial   = true;
    };

    class UITextOverlay : public UIComponent
    {
      public:
        UITextOverlay() = default;

        void AddText( std::string const &aText );

      protected:
        uint32_t             mLineCount = 0;
        std::list<sTextLine> mLines;
        std::string          mLeftOver;

        uint32_t mMaxLineCount = 1000;

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };
} // namespace SE::Core