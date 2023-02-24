#include "TextOverlay.h"
#include <sstream>

#include "Engine/Engine.h"

namespace SE::Core
{
    void UITextOverlay::PushStyles() {}
    void UITextOverlay::PopStyles() {}

    void UITextOverlay::AddText( std::string const &aText )
    {
        std::stringstream lStream( aText );
        std::string       lLine;
        std::string       lDelimiter = "\r\n";

        size_t lLineBeginPos    = 0;
        size_t lLineEndPosition = aText.find( lDelimiter, lLineBeginPos );
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
                if( lLastLine.mIsPartial )
                {
                    lLastLine.mLine += aText;
                }
            }
        }
        else
        {
            while( lLineEndPosition != std::string::npos )
            {
                std::string lLine = aText.substr( lLineBeginPos, lLineEndPosition );

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
                lLineEndPosition = aText.find( lDelimiter, lLineBeginPos );
            }
        }

        // while( std::getline( lStream, lLine, '\n' ) )
        // {
        //     if( !lLine.empty() && lLine.end() )

        //         if( mLines.empty() )
        //         {
        //             mLines.emplace_back( sTextLine{ 0, lLine, false } );
        //             mLineCount++;

        //             continue;
        //         }
        //     auto &lLastLine = mLines.back();

        //     if( lLastLine.mIsPartial )
        //     {
        //         lLastLine.mLine += lLine;
        //     }
        //     else
        //     {
        //         if( lLastLine.mLine == lLine )
        //         {
        //             lLastLine.mRepetitions += 1;
        //         }
        //         else
        //         {
        //             mLines.emplace_back( sTextLine{ 0, lLine, false } );

        //             if( mLineCount >= 25 )
        //                 mLines.pop_front();
        //             else
        //                 mLineCount++;
        //         }
        //     }
        // }
    }

    ImVec2 UITextOverlay::RequiredSize()
    {
        auto lTextSize = ImGui::CalcTextSize( "mText.c_str() " );

        return lTextSize;
    }

    void UITextOverlay::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        SE::Core::Engine::GetInstance()->UIContext()->PushFontFamily( FontFamilyFlags::MONO );
        ImGui::SetCursorPos( GetContentAlignedposition( mHAlign, mVAlign, aPosition, ImGui::CalcTextSize( "mText.c_str()" ), aSize ) );

        for( auto const &lLine : mLines ) ImGui::Text( lLine.mLine.c_str() );

        SE::Core::Engine::GetInstance()->UIContext()->PopFont();
    }

} // namespace SE::Core