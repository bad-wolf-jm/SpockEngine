#pragma once

#include "Component.h"

#include <list>

namespace SE::Core
{
    struct sTextLine
    {
        uint32_t    mRepetitions = 0;
        string_t mLine        = "";
        bool        mIsPartial   = true;
    };

    class UITextOverlay : public UIComponent
    {
      public:
        UITextOverlay() = default;

        void AddText( string_t const &aText );
        void Clear();

      protected:
        uint32_t             mLineCount = 0;
        std::list<sTextLine> mLines;
        string_t          mLeftOver;

        uint32_t mMaxLineCount = 100000;

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );

      public:
        static void *UITextOverlay_Create();
        static void  UITextOverlay_Destroy( void *aInstance );
        static void  UITextOverlay_AddText( void *aInstance, void *aText );
        static void  UITextOverlay_Clear( void *aInstance );
    };
} // namespace SE::Core