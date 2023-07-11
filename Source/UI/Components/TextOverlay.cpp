#include "TextOverlay.h"
#include <sstream>

#include "Engine/Engine.h"

namespace SE::Core
{

    void UITextOverlay::PushStyles() {}
    void UITextOverlay::PopStyles() {}

    void UITextOverlay::AddText( char *aBytes, int32_t aOffset, int32_t aCount )
    {
        SE::Logging::Info( "TEXT::: {} - {}", aOffset, aCount );
    }

    void UITextOverlay::AddText( string_t const &aText )
    {
        string_t lText = mLeftOver + aText;
        string_t lLine;
        string_t lDelimiter = "\r\n";

        size_t lLineBeginPos    = 0;
        size_t lLineEndPosition = lText.find( lDelimiter, lLineBeginPos );
        if( lLineEndPosition == string_t::npos )
        {
            if( mLines.empty() )
            {
                mLines.emplace_back( sTextLine{ 0, lLine, true } );
                mLineCount++;
            }
            else
            {
                auto &lLastLine = mLines.back();
                if( lLastLine.mIsPartial ) lLastLine.mLine += lText;
            }
        }
        else
        {
            while( lLineEndPosition != string_t::npos )
            {
                string_t lLine = lText.substr( lLineBeginPos, lLineEndPosition - lLineBeginPos );

                if( mLines.empty() )
                {
                    mLines.emplace_back( sTextLine{ 0, lLine, true } );
                    mLineCount++;
                }
                else
                {
                    auto &lLastLine = mLines.back();
                    if( lLastLine.mIsPartial )
                    {
                        lLastLine.mLine += lLine;
                        lLastLine.mIsPartial = false;
                    }
                    else
                    {
                        if( lLastLine.mLine == lLine )
                        {
                            lLastLine.mRepetitions += 1;
                        }
                        else
                        {
                            mLines.emplace_back( sTextLine{ 0, lLine, false } );

                            if( mLineCount >= mMaxLineCount )
                                mLines.pop_front();
                            else
                                mLineCount++;
                        }
                    }
                }
                lLineBeginPos    = lLineEndPosition + lDelimiter.length();
                lLineEndPosition = lText.find( lDelimiter, lLineBeginPos );
            }

            mLeftOver = lText.substr( lLineBeginPos );
        }
    }

    void UITextOverlay::Clear() { mLines.clear(); }

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

            SE::Logging::Info( "CONSOLE_SIZE=({}, {})", mConsoleWidth, mConsoleHeight );
        }

        if( !mIsVisible ) return;

        auto lDrawList = ImGui::GetWindowDrawList();

        ImGui::SetCursorPos( aPosition );

        ImGui::PushID( (void *)this );
        ImGui::PushStyleColor( ImGuiCol_ChildBg, ImVec4{ 0.01f, 0.01f, 0.01f, .9f } );
        ImGui::BeginChild( "##TextOverlay", aSize );

        auto lScreenPosition = ImGui::GetCursorScreenPos();

        SE::Core::Engine::GetInstance()->UIContext()->PushFontFamily( FontFamilyFlags::MONOSPACE );
        auto lHeight = ImGui::GetFontSize();
        auto lRadius = lHeight * 0.5f;
        SE::Core::Engine::GetInstance()->UIContext()->PopFont();
        auto lTagTextSize = ImGui::CalcTextSize( "9999" );
        auto lTagWidth    = lTagTextSize.x + lRadius * 2.0f;

        for( auto const &lLine : mLines )
        {
            auto lScreenPosition = ImGui::GetCursorScreenPos();
            lDrawList->AddRectFilled( lScreenPosition, lScreenPosition + ImVec2{ lTagWidth, lHeight }, ImColor( 32, 32, 32, 128 ),
                                      lRadius );

            auto   lTagText         = fmt::format( "{}", lLine.mRepetitions );
            auto   lLinePosition    = ImGui::GetCursorPos();
            auto   lTagTextRealSize = ImGui::CalcTextSize( lTagText.c_str() );
            ImVec2 lTagTextposition = ( ImVec2{ lTagWidth, lHeight } - lTagTextRealSize ) * 0.5f;

            ImGui::PushStyleColor( ImGuiCol_Text, ImVec4{ 0.9f, 0.9f, 0.9f, .4f } );
            ImGui::SetCursorPos( lLinePosition + lTagTextposition );
            ImGui::Text( lTagText.c_str() );
            ImGui::PopStyleColor();

            ImGui::SetCursorPos( lLinePosition + ImVec2{ lTagWidth + 5.0f, 0.0f } );

            SE::Core::Engine::GetInstance()->UIContext()->PushFontFamily( FontFamilyFlags::MONOSPACE );
            ImGui::Text( lLine.mLine.c_str() );
            SE::Core::Engine::GetInstance()->UIContext()->PopFont();
        }

        ImGui::EndChild();
        ImGui::PopStyleColor();
        ImGui::PopID();
    }
} // namespace SE::Core