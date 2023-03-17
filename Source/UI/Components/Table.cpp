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

    void UITable::OnRowClicked( std::function<void( uint32_t )> const &aOnRowClicked ) { mOnRowClicked = aOnRowClicked; }

    void UITable::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        ImGui::SetCursorPos( aPosition );

        const ImGuiTableFlags lTableFlags = ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY |
                                            ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_Resizable |
                                            ImGuiTableFlags_RowBg;

        if( ImGui::BeginTable( "##", mColumns.size(), lTableFlags, aSize ) )
        {
            ImGui::TableSetupScrollFreeze( 0, 1 );
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
                    if( mHoveredRow == lRow )
                        ImGui::TableSetBgColor( ImGuiTableBgTarget_RowBg0, IM_COL32(0x51, 0x08, 0x7E, 200) );
                    else if( mRowBackgroundColor.size() > 0 )
                        ImGui::TableSetBgColor( ImGuiTableBgTarget_RowBg0, mRowBackgroundColor[lRow] );

                    int lColumn = 0;
                    for( const auto &lColumnData : mColumns )
                    {
                        ImGui::TableSetColumnIndex( lColumn );

                        float lWidth = lThisTable->Columns[lColumn].WorkMaxX - lThisTable->Columns[lColumn].WorkMinX;

                        if( lThisTable->RowCellDataCurrent < 0 ||
                            lThisTable->RowCellData[lThisTable->RowCellDataCurrent].Column != lThisTable->CurrentColumn )
                            lThisTable->RowCellDataCurrent++;
                        auto lBgColor = lThisTable->RowCellData[lThisTable->RowCellDataCurrent].BgColor;
                        if( lColumnData->mBackgroundColor.size() > 0 )
                            ImGui::TableSetBgColor( ImGuiTableBgTarget_CellBg, lColumnData->mBackgroundColor[lRow] );
                        lColumnData->Render( lRow, ImVec2{ lWidth, mRowHeight } );
                        if( ImGui::IsItemHovered() ) mHoveredRow = lRow;
                        if( mOnRowClicked && ImGui::IsItemClicked() ) mOnRowClicked( lRow );
                        if( lColumnData->mBackgroundColor.size() > 0 ) ImGui::TableSetBgColor( ImGuiTableBgTarget_CellBg, lBgColor );

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

        ImVec2 lPrevPos = ImGui::GetCursorPos();
        ImVec2 lPos     = ImGui::GetCursorPos() + ImVec2{ aSize.x - lTextSize.x, ( aSize.y - lTextSize.y ) * 0.5f };
        ImGui::SetCursorPos( lPos );

        if( mForegroundColor.size() > 0 && ( mForegroundColor[aRow] != 0u ) )
            ImGui::PushStyleColor( ImGuiCol_Text, mForegroundColor[aRow] );
        ImGui::Text( lText.c_str() );
        if( mForegroundColor.size() > 0 && ( mForegroundColor[aRow] != 0u ) ) ImGui::PopStyleColor();

        ImVec2 lNewPos = ImGui::GetCursorPos();
        lNewPos.y      = lPrevPos.y + aSize.y;
        ImGui::SetCursorPos( lNewPos );
    }

    sStringColumn::sStringColumn( std::string aHeader, float aInitialSize )
        : sTableColumn{ aHeader, aInitialSize }
    {
    }

    uint32_t sStringColumn::Size() { return mData.size(); }

    void sStringColumn::Render( int aRow, ImVec2 aSize )
    {
        auto const &lTextSize = ImGui::CalcTextSize( mData[aRow].c_str() );

        ImVec2 lPrevPos = ImGui::GetCursorPos();
        ImVec2 lPos     = ImGui::GetCursorPos() + ImVec2{ 0.0f, ( aSize.y - lTextSize.y ) * 0.5f };
        ImGui::SetCursorPos( lPos );

        if( ( mForegroundColor.size() > 0 ) && ( mForegroundColor[aRow] != 0u ) )
            ImGui::PushStyleColor( ImGuiCol_Text, mForegroundColor[aRow] );
        ImGui::Text( mData[aRow].c_str() );
        if( mForegroundColor.size() > 0 && ( mForegroundColor[aRow] != 0u ) ) ImGui::PopStyleColor();

        ImVec2 lNewPos = ImGui::GetCursorPos();
        lNewPos.y      = lPrevPos.y + aSize.y;
        ImGui::SetCursorPos( lNewPos );
    }

} // namespace SE::Core