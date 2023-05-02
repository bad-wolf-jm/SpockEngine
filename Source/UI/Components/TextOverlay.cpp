#include "TextOverlay.h"
#include <sstream>

#include "DotNet/Runtime.h"
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

    void *UITextOverlay::UITextOverlay_Create()
    {
        auto lNewLabel = new UITextOverlay();

        return static_cast<void *>( lNewLabel );
    }

    void UITextOverlay::UITextOverlay_Destroy( void *aInstance ) { delete static_cast<UITextOverlay *>( aInstance ); }

    void UITextOverlay::UITextOverlay_AddText( void *aInstance, void *aText )
    {
        auto lInstance = static_cast<UITextOverlay *>( aInstance );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

        lInstance->AddText( lString );
    }

    void UITextOverlay::UITextOverlay_Clear( void *aInstance )
    {
        auto lInstance = static_cast<UITextOverlay *>( aInstance );

        lInstance->Clear();
    }

} // namespace SE::Core