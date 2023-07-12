#pragma once

#include "Component.h"

#include <list>

namespace SE::Core
{
    struct sTextLine
    {
        uint32_t mRepetitions = 0;
        string_t mLine        = "";
        bool     mIsPartial   = true;
    };

    enum class eTextEncoding : int32_t
    {
        UTF16 = 0,
        UTF8  = 1,
        ASCII = 2
    };

    struct sChar
    {
        char mCharacter;
        char mWidth;

        sChar( char aCharacter, char aWidth )
            : mCharacter{ aCharacter }
            , mWidth{ aWidth }
        {
        }
    };

    class UITextOverlay : public UIComponent
    {
      public:
        UITextOverlay() = default;

        void AddText( string_t const &aText );
        void AddText( char *aBytes, int32_t aOffset, int32_t aCount );
        void Clear();

      protected:
        // uint32_t             mLineCount = 0;
        // std::list<sTextLine> mLines;
        // string_t             mLeftOver;

        uint32_t mMaxLineCount = 100000;

        uint32_t mCharWidth     = 0;
        uint32_t mCharHeight    = 0;
        uint32_t mConsoleWidth  = 0;
        uint32_t mConsoleHeight = 0;

        std::vector<sChar> mCharacters;
        eTextEncoding      mEncoding = eTextEncoding::UTF16;

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };
} // namespace SE::Core