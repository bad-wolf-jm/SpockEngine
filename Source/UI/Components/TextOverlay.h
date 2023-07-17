#pragma once

#include "Component.h"

#include <list>
#include <mutex>

namespace SE::Core
{
    enum class eTextEncoding : int32_t
    {
        UTF16 = 0,
        UTF8  = 1,
        ASCII = 2
    };

    struct sCharacter
    {
        char  mCharacter[4] = { 0 };
        char  mByteCount    = { 0 };
        char  mWidth        = '\0';
        float mCharWidth    = 0.0;
        // TODO: backgound color
        // TODO: foregound color

        sCharacter() = default;
        sCharacter( char aCharacter, char aWidth )
            : mCharacter{ aCharacter }
            , mWidth{ aWidth }
        {
        }
    };

    struct sLine
    {
        uint32_t mBegin;
        uint32_t mEnd;
    };

    class UITextOverlay : public UIComponent
    {
      public:
        UITextOverlay() = default;

        void AddText( string_t const &aText );
        void AddText( char *aBytes, int32_t aOffset, int32_t aCount );
        void Clear();

      protected:
        uint32_t mMaxLineCount = 100000;

        uint32_t mCharWidth     = 0;
        uint32_t mCharHeight    = 0;
        uint32_t mConsoleWidth  = 0;
        uint32_t mConsoleHeight = 0;

        float mLeftMargin   = 10.0f;
        float mRightMargin  = 10.0f;
        float mTopMargin    = 10.0f;
        float mBottomMargin = 10.0f;

        std::vector<sCharacter> mCharacters;
        std::vector<sLine>      mLines;
        std::mutex              mLinesMutex;
        eTextEncoding           mEncoding = eTextEncoding::UTF16;

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );
        void   Layout();
    };
} // namespace SE::Core