#include "Table.h"

namespace SE::Core
{

    void UITable::PushStyles() {}
    void UITable::PopStyles() {}

    void UITable::SetRowHeight( float aRowHeight ) { mRowHeight = aRowHeight; }

    void UITable::AddColumn( Ref<sTableColumn> aColumn ) { mColumns.push_back( aColumn ); }

    void UITable::SetData( Ref<sTableData> aData ) { mData = aData; }

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

            ImGuiListClipper lRowClipping;
            lRowClipping.Begin( mData->CountRows() );
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

                        float lWidth  = lThisTable->Columns[lColumn].ItemWidth;
                        float lHeight = 30.0f;

                        lColumnData->Render( mData->Get( lRow, lColumn ), ImVec2{ lWidth, lHeight } );
                    }
                }
            }
            ImGui::EndTable();
        }
    }

} // namespace SE::Core