#include "Table.h"

namespace SE::Core
{
    sTableColumn::sTableColumn( std::string aHeader, float aInitialSize )
        : mHeader{ aHeader }
        , mInitialSize{ aInitialSize }
    {
    }

    void UITable::PushStyles() {}
    void UITable::PopStyles() {}

    void UITable::SetRowHeight( float aRowHeight ) { mRowHeight = aRowHeight; }

    void UITable::AddColumn( Ref<sTableColumn> aColumn ) { mColumns.push_back( aColumn ); }

    ImVec2 UITable::RequiredSize() { return ImVec2{}; }

    void UITable::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        ImGui::SetCursorPos( aPosition );

        const ImGuiTableFlags lTableFlags = ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY |
                                            ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_Resizable |
                                            ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable;

        if( ImGui::BeginTable( "##", mColumns.size(), lTableFlags, aSize ) )
        {
            ImGuiTable *lThisTable = ImGui::GetCurrentContext()->CurrentTable;

            for( const auto &lColumn : mColumns )
                ImGui::TableSetupColumn( lColumn->mHeader.c_str(), ImGuiTableColumnFlags_None, lColumn->mInitialSize );

            ImGui::TableHeadersRow();

            auto lRowCount = std::numeric_limits<uint32_t>::max();
            for( const auto &lColumn : mColumns ) lRowCount = std::min( lRowCount, lColumn->Size() );

            ImGuiListClipper lRowClipping;
            lRowClipping.Begin( lRowCount );
            ImGui::TableNextRow();
            while( lRowClipping.Step() )
            {
                for( int lRow = lRowClipping.DisplayStart; lRow < lRowClipping.DisplayEnd; lRow++ )
                {
                    ImGui::TableNextRow();

                    int lColumn = 0;
                    for( const auto &lColumnData : mColumns )
                    {
                        ImGui::TableSetColumnIndex( lColumn );

                        float lWidth = lThisTable->Columns[lColumn].WorkMaxX - lThisTable->Columns[lColumn].WorkMinX;

                        lColumnData->Render( lRow, ImVec2{ lWidth, mRowHeight } );

                        lColumn++;
                    }
                }
            }
            ImGui::EndTable();
        }
    }

    sFloat64Column::sFloat64Column( std::string aHeader, float aInitialSize, std::string aFormat, std::string aNaNFormat )
        : sTableColumn{ aHeader, aInitialSize }
        , mFormat{ aFormat }
        , mNaNFormat{ aNaNFormat }
    {
    }

    uint32_t sFloat64Column::Size() { return mData.size(); }

    void sFloat64Column::Render( int aRow, ImVec2 aSize )
    {
        std::string lText;
        if( std::isnan( mData[aRow] ) )
            lText = fmt::format( mNaNFormat, mData[aRow] );
        else
            lText = fmt::format( mFormat, mData[aRow] );

        auto const &lTextSize = ImGui::CalcTextSize( lText.c_str() );

        ImVec2 lPos = ImGui::GetCursorPos() + ImVec2{ aSize.x - lTextSize.x, ( aSize.y - lTextSize.y ) * 0.5f };
        ImGui::SetCursorPos( lPos );
        ImGui::Text( lText.c_str() );
    }


    sStringColumn::sStringColumn( std::string aHeader, float aInitialSize )
        : sTableColumn{ aHeader, aInitialSize }
    {
    }

    uint32_t sStringColumn::Size() { return mData.size(); }

    void sStringColumn::Render( int aRow, ImVec2 aSize )
    {
        auto const &lTextSize = ImGui::CalcTextSize( mData[aRow].c_str() );

        // ImVec2 lPos = ImGui::GetCursorPos() + ImVec2{ aSize.x - lTextSize.x, ( aSize.y - lTextSize.y ) * 0.5f };
        // ImGui::SetCursorPos( lPos );
        ImGui::Text( mData[aRow].c_str() );
    }

} // namespace SE::Core