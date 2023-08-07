#include "Table.h"

namespace SE::Core
{
    UITableColumn::UITableColumn( string_t aHeader, float aInitialSize )
        : mHeader{ aHeader }
        , mInitialSize{ aInitialSize }
    {
        SetAlignment( eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        SE::Logging::Info( "{}", (int)mHAlign );
    }

    void UITableColumn::Clear()
    {
        mForegroundColor.clear();
        mBackgroundColor.clear();
    }

    // void UITableColumn::PushStyles() {}
    // void UITableColumn::PopStyles() {}

    void UITable::PushStyles()
    {
    }
    void UITable::PopStyles()
    {
    }

    void UITable::SetRowHeight( float aRowHeight )
    {
        mRowHeight = aRowHeight;
    }

    void UITable::AddColumn( ref_t<UITableColumn> aColumn )
    {
        mColumns.push_back( aColumn.get() );
    }
    void UITable::AddColumn( UITableColumn *aColumn )
    {
        mColumns.push_back( aColumn );
    }

    ImVec2 UITable::RequiredSize()
    {
        return ImVec2{};
    }

    void UITable::OnRowClicked( std::function<void( uint32_t )> const &aOnRowClicked )
    {
        mOnRowClicked = aOnRowClicked;
    }

    void UITable::DrawTableCell( ImGuiTable *lThisTable, UITableColumn *lColumnData, int lColumn, int lRow )
    {
        float lWidth = lThisTable->Columns[lColumn].WorkMaxX - lThisTable->Columns[lColumn].WorkMinX;

        if( lThisTable->RowCellDataCurrent < 0 ||
            lThisTable->RowCellData[lThisTable->RowCellDataCurrent].Column != lThisTable->CurrentColumn )
            lThisTable->RowCellDataCurrent++;

        auto lBgColor = lThisTable->RowCellData[lThisTable->RowCellDataCurrent].BgColor;
        if( lColumnData->mBackgroundColor.size() > 0 )
            ImGui::TableSetBgColor( ImGuiTableBgTarget_CellBg, lColumnData->mBackgroundColor[lRow] );

        auto lPos = ImGui::GetCursorPos();
        ImGui::Dummy( ImVec2{ lWidth, mRowHeight } );
        if( ImGui::IsItemHovered() )
        {
            mHoveredRow = lRow;

            if( lColumnData->mToolTip.size() > 0 )
            {
                ImGui::BeginTooltip();
                lColumnData->mToolTip[lRow]->Update( ImVec2{}, lColumnData->mToolTip[lRow]->RequiredSize() );
                ImGui::EndTooltip();
            }
        }

        if( ImGui::IsItemClicked() )
            mSelectedRow = lRow;

        ImGui::SetCursorPos( lPos );
        lColumnData->Render( lRow, ImVec2{ lWidth, mRowHeight } );

        if( mOnRowClicked && ImGui::IsItemClicked() )
            mOnRowClicked( lRow );

        if( lColumnData->mBackgroundColor.size() > 0 )
            ImGui::TableSetBgColor( ImGuiTableBgTarget_CellBg, lBgColor );
    }

    void UITable::DrawTableRows( ImGuiTable *lThisTable, uint32_t lRowCount, int aRowStart, int aRowEnd )
    {
        for( int lRowID = aRowStart; lRowID < aRowEnd; lRowID++ )
        {
            int lRow = lRowID;
            if( mDisplayedRowIndices.has_value() )
                lRow = mDisplayedRowIndices.value()[lRowID];

            if( lRow > lRowCount )
                continue;

            ImGui::TableNextRow();

            if( mSelectedRow == lRow )
                ImGui::TableSetBgColor( ImGuiTableBgTarget_RowBg0, IM_COL32( 1, 50, 32, 128 ) );
            else if( mHoveredRow == lRow )
                ImGui::TableSetBgColor( ImGuiTableBgTarget_RowBg0, IM_COL32( 0x51, 0x08, 0x7E, 128 ) );
            else if( mRowBackgroundColor.size() > 0 )
                ImGui::TableSetBgColor( ImGuiTableBgTarget_RowBg0, mRowBackgroundColor[lRow] );

            int lColumn = 0;
            for( const auto &lColumnData : mColumns )
            {
                ImGui::TableSetColumnIndex( lColumn );

                DrawTableCell(lThisTable, lColumnData, lColumn, lRow);

                lColumn++;
            }
        }
    }

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
            for( const auto &lColumn : mColumns )
                lRowCount = std::min( lRowCount, lColumn->Size() );

            int lDisplayedRowCount = lRowCount;
            if( mDisplayedRowIndices.has_value() )
                lDisplayedRowCount = mDisplayedRowIndices.value().size();

            ImGuiListClipper lRowClipping;
            lRowClipping.Begin( lDisplayedRowCount );
            while( lRowClipping.Step() )
            {
                DrawTableRows( lThisTable, lRowCount, lRowClipping.DisplayStart, lRowClipping.DisplayEnd );
            }
            ImGui::EndTable();
        }
    }

    UIStringColumn::UIStringColumn( string_t aHeader, float aInitialSize )
        : UITableColumn{ aHeader, aInitialSize }
    {
    }

    uint32_t UIStringColumn::Size()
    {
        return mData.size();
    }

    void UIStringColumn::Render( int aRow, ImVec2 aSize )
    {
        ImVec2      lPrevPos  = ImGui::GetCursorPos();
        auto const &lTextSize = ImGui::CalcTextSize( mData[aRow].c_str() );
        auto        lPos      = GetContentAlignedposition( mHAlign, mVAlign, ImGui::GetCursorPos(), lTextSize, aSize );

        ImGui::SetCursorPos( lPos );

        if( ( mForegroundColor.size() > 0 ) && ( mForegroundColor[aRow] != 0u ) )
            ImGui::PushStyleColor( ImGuiCol_Text, mForegroundColor[aRow] );

        ImGui::Text( mData[aRow].c_str() );

        if( mForegroundColor.size() > 0 && ( mForegroundColor[aRow] != 0u ) )
            ImGui::PopStyleColor();

        ImVec2 lNewPos = ImGui::GetCursorPos();
        lNewPos.y      = lPrevPos.y + aSize.y;
        ImGui::SetCursorPos( lNewPos );
    }

    void UIStringColumn::Clear()
    {
        UITableColumn::Clear();
        mData.clear();
    }

    ImVec2 UIStringColumn::RequiredSize()
    {
        return ImVec2{};
    }

    void UIStringColumn::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
    }

} // namespace SE::Core