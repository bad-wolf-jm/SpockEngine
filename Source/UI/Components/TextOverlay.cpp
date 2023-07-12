#include "TextOverlay.h"

#include <codecvt>
#include <locale>
#include <sstream>

#include "Engine/Engine.h"

namespace SE::Core
{

    void UITextOverlay::PushStyles() {}
    void UITextOverlay::PopStyles() {}

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
                int   lBytesNeeded = WideCharToMultiByte( CP_UTF8, 0, (wchar_t *)&aCharArray[i], 1, NULL, 0, NULL, NULL );
                auto &lChar        = mCharacters.emplace_back();
                lChar.mByteCount   = lBytesNeeded;
                WideCharToMultiByte( CP_UTF8, 0, (wchar_t *)&aCharArray[i], 1, (LPSTR)&lChar.mCharacter, lBytesNeeded, NULL, NULL );
                i += 2;
            }
            break;
            case eTextEncoding::UTF8: break;
            case eTextEncoding::ASCII:
            {
                auto &lChar         = mCharacters.emplace_back();
                lChar.mByteCount    = 1;
                lChar.mCharacter[0] = aCharArray[i];
                i++;
            }
            break;
            default: break;
            }
        }
    }

    void UITextOverlay::AddText( string_t const &aText ) {}

    void UITextOverlay::Clear() { mCharacters.clear(); }

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

    void UITextOverlay::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        if( mCharWidth == 0 )
        {
            SE::Core::Engine::GetInstance()->UIContext()->PushFontFamily( FontFamilyFlags::MONOSPACE );
            auto lCharSize = ImGui::CalcTextSize( "M" );
            SE::Core::Engine::GetInstance()->UIContext()->PopFont();

            mCharWidth  = lCharSize.x;
            mCharHeight = lCharSize.y;
        }

        auto lNewConsoleWidth  = static_cast<uint32_t>( aSize.x / mCharWidth );
        auto lNewConsoleHeight = static_cast<uint32_t>( aSize.y / mCharHeight );

        if( ( lNewConsoleWidth != mConsoleWidth ) || ( lNewConsoleHeight != mConsoleHeight ) )
        {
            mConsoleWidth  = lNewConsoleWidth;
            mConsoleHeight = lNewConsoleHeight;
        }

        if( !mIsVisible ) return;

        ImGui::PushID( (void *)this );
        ImGui::PushStyleColor( ImGuiCol_ChildBg, ImVec4{ 0.01f, 0.01f, 0.01f, .9f } );
        ImGui::BeginChild( "##TextOverlay", aSize );

        auto  *g               = ImGui::GetCurrentContext();
        auto  *lDrawlist       = ImGui::GetWindowDrawList();
        ImVec2 lCursorPosition = ImGui::GetCursorScreenPos() + aPosition;
        float  lTopPosition    = lCursorPosition.y;
        float  lLeftPosition   = lCursorPosition.x;

        int32_t lLineCursorPosition = 0;
        SE::Core::Engine::GetInstance()->UIContext()->PushFontFamily( FontFamilyFlags::MONOSPACE );
        for( int32_t i = 0; i < mCharacters.size(); i++ )
        {
            if( mCharacters[i].mCharacter[0] == '\n' )
            {
                lLineCursorPosition = 0;

                lCursorPosition.x = lLeftPosition;
                lCursorPosition.y += mCharHeight;
                continue;
            }

            lDrawlist->AddText( g->Font, g->FontSize, lCursorPosition, ImGui::GetColorU32( ImGuiCol_Text ), mCharacters[i].mCharacter,
                                mCharacters[i].mCharacter + 1 );

            if( lLineCursorPosition >= mConsoleWidth )
            {
                lLineCursorPosition = 0;

                lCursorPosition.x = lLeftPosition;
                lCursorPosition.y += mCharHeight;
            }
            else
            {
                lLineCursorPosition++;

                lCursorPosition.x += mCharWidth;
            }
        }
        SE::Core::Engine::GetInstance()->UIContext()->PopFont();

        ImGui::ItemSize( ImVec2{ aSize.x, ( lCursorPosition.y + mCharHeight ) - lTopPosition }, 0.0f );
        ImGui::EndChild();
        ImGui::PopStyleColor();
        ImGui::PopID();
    }
} // namespace SE::Core