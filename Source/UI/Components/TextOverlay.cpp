#include "TextOverlay.h"

#include <codecvt>
#include <locale>
#include <sstream>

#include "Engine/Engine.h"

namespace SE::Core
{

    void UITextOverlay::PushStyles()
    {
    }
    void UITextOverlay::PopStyles()
    {
    }

    static void FillUtf8Utf16( sCharacter &aOut, char *aIn )
    {
        aOut.mByteCount = WideCharToMultiByte( CP_UTF8, 0, (wchar_t *)&aIn[0], 1, NULL, 0, NULL, NULL );
        WideCharToMultiByte( CP_UTF8, 0, (wchar_t *)&aIn[0], 1, (LPSTR)&aOut.mCharacter, aOut.mByteCount, NULL, NULL );
    }

    static void FillUtf8Utf8( sCharacter &aOut, char *aIn )
    {
        char aInByte = aIn[0];

        if( aInByte < 0x80u )
        {
            aOut.mCharacter[0] = aIn[0];

            aOut.mByteCount = 1;
        }
        else if( aInByte < 0xe0u )
        {
            aOut.mCharacter[0] = aIn[0];
            aOut.mCharacter[1] = aIn[1];

            aOut.mByteCount = 2;
        }
        else if( aInByte < 0xf0u )
        {
            aOut.mCharacter[0] = aIn[0];
            aOut.mCharacter[1] = aIn[1];
            aOut.mCharacter[2] = aIn[2];

            aOut.mByteCount = 3;
        }
        else if( aInByte < 0xf8u )
        {
            aOut.mCharacter[0] = aIn[0];
            aOut.mCharacter[1] = aIn[1];
            aOut.mCharacter[2] = aIn[2];
            aOut.mCharacter[3] = aIn[3];

            aOut.mByteCount = 4;
        }
        else
        {
            aOut.mByteCount = 1;
        }
    }

    void UITextOverlay::AddText( char *aBytes, int32_t aOffset, int32_t aCount )
    {
        char *aCharArray = aBytes + aOffset;

        int32_t i = 0;
        while( i < aCount )
        {
            switch( mEncoding )
            {
            case eTextEncoding::UTF16:
            {
                auto &lChar = mCharacters.emplace_back();

                FillUtf8Utf16( lChar, &aCharArray[i] );
                lChar.mCharWidth = mCharWidth;
                i += 2;
            }
            break;
            case eTextEncoding::UTF8:
            {
                auto &lChar = mCharacters.emplace_back();

                FillUtf8Utf8( lChar, &aCharArray[i] );
                lChar.mCharWidth = mCharWidth;
                i += lChar.mByteCount;
            }
            break;
            case eTextEncoding::ASCII:
            {
                auto &lChar = mCharacters.emplace_back();

                lChar.mByteCount    = 1;
                lChar.mCharacter[0] = aCharArray[i];
                lChar.mCharWidth    = mCharWidth;
                i++;
            }
            break;
            default:
                break;
            }
        }

        std::lock_guard<std::mutex> guard( mLinesMutex );
        Layout();
    }

    void UITextOverlay::AddText( string_t const &aText )
    {
    }

    void UITextOverlay::Clear()
    {
        mCharacters.clear();
        mLines.clear();
    }

    ImVec2 UITextOverlay::RequiredSize()
    {
        SE::Core::Engine::GetInstance()->UIContext()->PushFontFamily( FontFamilyFlags::MONOSPACE );
        auto lHeight = ImGui::GetFontSize();
        auto lRadius = lHeight * 0.5f;
        SE::Core::Engine::GetInstance()->UIContext()->PopFont();
        auto lTagTextSize = ImGui::CalcTextSize( "9999" );
        auto lTagWidth    = lTagTextSize.x + lRadius * 2.0f;

        return ImVec2{ lTagWidth, lHeight };
    }

    void UITextOverlay::Layout()
    {
        mLines.clear();

        uint32_t lLineStart                   = 0;
        int32_t  lCursorPositionInCurrentLine = 0;

        size_t lCharacterCount = mCharacters.size();
        for( int32_t i = 0; i < lCharacterCount; i++ )
        {
            if( mCharacters[i].mCharacter[0] == '\n' )
            {
                auto &lLineData  = mLines.emplace_back();
                lLineData.mBegin = lLineStart;
                lLineData.mEnd   = i - 1;
                lLineStart       = i + 1;

                lCursorPositionInCurrentLine = 0;

                continue;
            }

            if( lCursorPositionInCurrentLine >= mConsoleWidth )
            {
                auto &lLineData  = mLines.emplace_back();
                lLineData.mBegin = lLineStart;
                lLineData.mEnd   = i - 1;
                lLineStart       = i;

                lCursorPositionInCurrentLine = 0;
            }
            else
            {
                lCursorPositionInCurrentLine++;
            }
        }
    }

    void UITextOverlay::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        std::lock_guard<std::mutex> guard( mLinesMutex );

        if( mCharWidth == 0 )
        {
            SE::Core::Engine::GetInstance()->UIContext()->PushFontFamily( FontFamilyFlags::MONOSPACE );
            auto lCharSize = ImGui::CalcTextSize( "M" );
            SE::Core::Engine::GetInstance()->UIContext()->PopFont();

            mCharWidth  = lCharSize.x;
            mCharHeight = lCharSize.y;
        }

        float lEffectiveWidth  = aSize.x - ( mLeftMargin + mRightMargin );
        float lEffectiveHeight = aSize.y - ( mTopMargin + mBottomMargin );

        auto lNewConsoleWidth  = static_cast<uint32_t>( lEffectiveWidth / mCharWidth );
        auto lNewConsoleHeight = static_cast<uint32_t>( lEffectiveHeight / mCharHeight );

        if( ( lNewConsoleWidth != mConsoleWidth ) || ( lNewConsoleHeight != mConsoleHeight ) )
        {
            mConsoleWidth  = lNewConsoleWidth;
            mConsoleHeight = lNewConsoleHeight;

            Layout();
        }

        if( !mIsVisible )
            return;
        if( mLines.size() == 0 )
            return;

        ImGui::PushID( (void *)this );
        ImGui::PushStyleColor( ImGuiCol_ChildBg, ImVec4{ 0.01f, 0.01f, 0.01f, .9f } );
        ImGui::BeginChild( "##TextOverlay", aSize );

        auto  lScrollY  = ImGui::GetScrollY();
        auto *g         = ImGui::GetCurrentContext();
        auto *lDrawlist = ImGui::GetWindowDrawList();

        SE::Core::Engine::GetInstance()->UIContext()->PushFontFamily( FontFamilyFlags::MONOSPACE );

        uint32_t lFirstLine = 0;
        if( mLines.size() >= mConsoleHeight )
            lFirstLine = mLines.size() - mConsoleHeight - 1;

        ImGui::SetScrollY( lFirstLine * mCharHeight );

        ImGuiListClipper lTextClipper;
        lTextClipper.Begin( mLines.size() );
        while( lTextClipper.Step() )
        {
            for( int l = lTextClipper.DisplayStart; l < lTextClipper.DisplayEnd; l++ )
            {
                auto lLine = mLines[l];

                ImVec2 lCursorPosition = ImGui::GetCursorScreenPos() + aPosition + ImVec2{ mLeftMargin, mTopMargin };
                float  lLeftPosition   = lCursorPosition.x;
                for( uint32_t i = lLine.mBegin; i < lLine.mEnd; i++ )
                {
                    if( mCharacters[i].mBackground )
                        lDrawlist->AddRectFilled( lCursorPosition, lCursorPosition + ImVec2{ (float)mCharWidth, (float)mCharHeight },
                                                  mCharacters[i].mBackground, 0.0f, 0 );

                    uint32_t lForegroundColor =
                        mCharacters[i].mForeground == 0 ? ImGui::GetColorU32( ImGuiCol_Text ) : mCharacters[i].mForeground;
                    lDrawlist->AddText( g->Font, g->FontSize, lCursorPosition, lForegroundColor, mCharacters[i].mCharacter,
                                        mCharacters[i].mCharacter + 1 );

                    lCursorPosition.x += mCharWidth;
                }

                lCursorPosition.x = lLeftPosition;
                lCursorPosition.y += mCharHeight;

                ImGui::SetCursorPosY( ImGui::GetCursorPosY() + mCharHeight );
            }
        }
        ImGui::SetCursorPosY( ImGui::GetCursorPosY() + mCharHeight );

        SE::Core::Engine::GetInstance()->UIContext()->PopFont();
        ImGui::EndChild();
        ImGui::PopStyleColor();
        ImGui::PopID();
    }
} // namespace SE::Core