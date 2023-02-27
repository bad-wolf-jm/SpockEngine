#include "TextOverlay.h"
#include <sstream>

#include "Engine/Engine.h"

namespace SE::Core
{
    void UITextOverlay::PushStyles() {}
    void UITextOverlay::PopStyles() {}

    void UITextOverlay::AddText( std::string const &aText )
    {
        std::string lText = mLeftOver + aText;
        std::string lLine;
        std::string lDelimiter = "\r\n";

        size_t lLineBeginPos    = 0;
        size_t lLineEndPosition = lText.find( lDelimiter, lLineBeginPos );
        if( lLineEndPosition == std::string::npos )
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
            while( lLineEndPosition != std::string::npos )
            {
                std::string lLine = lText.substr( lLineBeginPos, lLineEndPosition - lLineBeginPos );

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

                            if( mLineCount >= 25 )
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

    ImVec2 UITextOverlay::RequiredSize()
    {
        auto lTextSize = ImGui::CalcTextSize( "mText.c_str() " );

        return lTextSize;
    }

    void UITextOverlay::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        auto lDrawList = ImGui::GetWindowDrawList();
        ImGui::SetCursorPos( GetContentAlignedposition( mHAlign, mVAlign, aPosition, ImGui::CalcTextSize( "mText.c_str()" ), aSize ) );

        SE::Core::Engine::GetInstance()->UIContext()->PushFontFamily( FontFamilyFlags::MONO );
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

            auto   lLinePosition    = ImGui::GetCursorPos();
            auto   lTagText         = fmt::format( "{}", lLine.mRepetitions );
            auto   lTagTextRealSize = ImGui::CalcTextSize( lTagText.c_str() );
            ImVec2 lTagTextposition = ( ImVec2{ lTagWidth, lHeight } - lTagTextRealSize ) * 0.5f;

            ImGui::PushStyleColor( ImGuiCol_Text, ImVec4{ 0.9f, 0.9f, 0.9f, .4f } );
            ImGui::SetCursorPos( lLinePosition + lTagTextposition );
            ImGui::Text( lTagText.c_str() );
            ImGui::PopStyleColor();

            ImGui::SetCursorPos( lLinePosition + ImVec2{ lTagWidth + 5.0f, 0.0f } );

            SE::Core::Engine::GetInstance()->UIContext()->PushFontFamily( FontFamilyFlags::MONO );
            ImGui::Text( lLine.mLine.c_str() );
            SE::Core::Engine::GetInstance()->UIContext()->PopFont();
        }
    }

} // namespace SE::Core