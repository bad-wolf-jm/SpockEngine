#include "Table.h"

#include "DotNet/Runtime.h"

namespace SE::Core
{
    sTableColumn::sTableColumn( std::string aHeader, float aInitialSize )
        : mHeader{ aHeader }
        , mInitialSize{ aInitialSize }
    {
    }

    void sTableColumn::Clear()
    {
        mForegroundColor.clear();
        mBackgroundColor.clear();
    }

    void UITable::PushStyles() {}
    void UITable::PopStyles() {}

    void UITable::SetRowHeight( float aRowHeight ) { mRowHeight = aRowHeight; }

    void UITable::AddColumn( Ref<sTableColumn> aColumn ) { mColumns.push_back( aColumn.get() ); }
    void UITable::AddColumn( sTableColumn *aColumn ) { mColumns.push_back( aColumn ); }

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
            while( lRowClipping.Step() )
            {
                for( int lRow = lRowClipping.DisplayStart; lRow < lRowClipping.DisplayEnd; lRow++ )
                {
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

                        float lWidth = lThisTable->Columns[lColumn].WorkMaxX - lThisTable->Columns[lColumn].WorkMinX;

                        if( lThisTable->RowCellDataCurrent < 0 ||
                            lThisTable->RowCellData[lThisTable->RowCellDataCurrent].Column != lThisTable->CurrentColumn )
                            lThisTable->RowCellDataCurrent++;
                        auto lBgColor = lThisTable->RowCellData[lThisTable->RowCellDataCurrent].BgColor;
                        if( lColumnData->mBackgroundColor.size() > 0 )
                            ImGui::TableSetBgColor( ImGuiTableBgTarget_CellBg, lColumnData->mBackgroundColor[lRow] );
                        auto lPos = ImGui::GetCursorPos();
                        ImGui::Dummy( ImVec2{ lWidth, mRowHeight } );
                        if( ImGui::IsItemHovered() ) mHoveredRow = lRow;
                        if( ImGui::IsItemClicked() ) mSelectedRow = lRow;
                        ImGui::SetCursorPos( lPos );
                        lColumnData->Render( lRow, ImVec2{ lWidth, mRowHeight } );

                        if( mOnRowClicked && ImGui::IsItemClicked() ) mOnRowClicked( lRow );
                        if( lColumnData->mBackgroundColor.size() > 0 ) ImGui::TableSetBgColor( ImGuiTableBgTarget_CellBg, lBgColor );

                        lColumn++;
                    }
                }
            }
            ImGui::EndTable();
        }
    }

    void *UITable::UITable_Create()
    {
        auto lNewTable = new UITable();

        return static_cast<void *>( lNewTable );
    }

    void UITable::UITable_Destroy( void *aSelf ) { delete static_cast<UITable *>( aSelf ); }

    void UITable::UITable_OnRowClicked( void *aInstance, void *aDelegate )
    {
        auto lInstance = static_cast<UITable *>( aInstance );
        auto lDelegate = static_cast<MonoObject *>( aDelegate );

        if( lInstance->mOnRowClickDelegate != nullptr ) mono_gchandle_free( lInstance->mOnRowClickDelegateHandle );

        lInstance->mOnRowClickDelegate       = aDelegate;
        lInstance->mOnRowClickDelegateHandle = mono_gchandle_new( static_cast<MonoObject *>( aDelegate ), true );

        lInstance->OnRowClicked(
            [lInstance, lDelegate]( int aValue )
            {
                auto lDelegateClass = mono_object_get_class( lDelegate );
                auto lInvokeMethod  = mono_get_delegate_invoke( lDelegateClass );

                void *lParams[] = { (void *)&aValue };
                auto  lValue    = mono_runtime_invoke( lInvokeMethod, lDelegate, lParams, nullptr );
            } );
    }

    void UITable::UITable_AddColumn( void *aSelf, void *aColumn )
    {
        auto lInstance = static_cast<UITable *>( aSelf );
        auto lColumn   = static_cast<sTableColumn *>( aColumn );

        lInstance->AddColumn( lColumn );
    }

    void UITable::UITable_SetRowHeight( void *aSelf, float aRowHeight )
    {
        auto lInstance = static_cast<UITable *>( aSelf );

        lInstance->SetRowHeight( aRowHeight );
    }

    void UITable::UITable_ClearRowBackgroundColor( void *aSelf )
    {
        auto lSelf = static_cast<UITable *>( aSelf );

        lSelf->mRowBackgroundColor.clear();
    }

    void UITable::UITable_SetRowBackgroundColor( void *aSelf, void *aValue )
    {
        auto lSelf = static_cast<UITable *>( aSelf );

        lSelf->mRowBackgroundColor.clear();
        for( auto &x : DotNetRuntime::AsVector<ImVec4>( static_cast<MonoObject *>( aValue ) ) )
            lSelf->mRowBackgroundColor.push_back( ImColor( x ) );
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

    void sFloat64Column::Clear()
    {
        sTableColumn::Clear();
        mData.clear();
    }

    void *sFloat64Column::UIFloat64Column_Create()
    {
        auto lNewColumn = new sFloat64Column();

        return static_cast<void *>( lNewColumn );
    }

    void *sFloat64Column::UIFloat64Column_CreateFull( void *aHeader, float aInitialSize, void *aFormat, void *aNaNFormat )
    {
        auto lHeader    = DotNetRuntime::NewString( static_cast<MonoString *>( aHeader ) );
        auto lFormat    = DotNetRuntime::NewString( static_cast<MonoString *>( aFormat ) );
        auto lNaNFormat = DotNetRuntime::NewString( static_cast<MonoString *>( aNaNFormat ) );
        auto lNewColumn = new sFloat64Column( lHeader, aInitialSize, lFormat, lNaNFormat );

        return static_cast<void *>( lNewColumn );
    }

    void sFloat64Column::UIFloat64Column_Destroy( void *aSelf ) { delete static_cast<sFloat64Column *>( aSelf ); }

    void sFloat64Column::UIFloat64Column_Clear( void *aSelf )
    {
        auto lSelf = static_cast<sFloat64Column *>( aSelf );

        lSelf->Clear();
    }

    void sFloat64Column::UIFloat64Column_SetData( void *aSelf, void *aValue )
    {
        auto lSelf = static_cast<sFloat64Column *>( aSelf );

        lSelf->mData = DotNetRuntime::AsVector<double>( static_cast<MonoObject *>( aValue ) );
    }

    void sFloat64Column::UIFloat64Column_SetDataWithForegroundColor( void *aSelf, void *aValue, void *aForegroundColor )
    {
        auto lSelf = static_cast<sFloat64Column *>( aSelf );

        lSelf->mData = DotNetRuntime::AsVector<double>( static_cast<MonoObject *>( aValue ) );
        lSelf->mForegroundColor.clear();
        for( auto const &x : DotNetRuntime::AsVector<ImVec4>( static_cast<MonoObject *>( aForegroundColor ) ) )
            lSelf->mForegroundColor.push_back( ImColor( x ) );
    }

    void sFloat64Column::UIFloat64Column_SetDataWithForegroundAndBackgroundColor( void *aSelf, void *aValue, void *aForegroundColor,
                                                                                  void *aBackroundColor )
    {
        auto lSelf = static_cast<sFloat64Column *>( aSelf );

        lSelf->mData = DotNetRuntime::AsVector<double>( static_cast<MonoObject *>( aValue ) );

        lSelf->mForegroundColor.clear();
        for( auto const &x : DotNetRuntime::AsVector<ImVec4>( static_cast<MonoObject *>( aForegroundColor ) ) )
            lSelf->mForegroundColor.push_back( ImColor( x ) );

        lSelf->mBackgroundColor.clear();
        for( auto const &x : DotNetRuntime::AsVector<ImVec4>( static_cast<MonoObject *>( aBackroundColor ) ) )
            lSelf->mBackgroundColor.push_back( ImColor( x ) );
    }

    sUint32Column::sUint32Column( std::string aHeader, float aInitialSize )
        : sTableColumn{ aHeader, aInitialSize }
    {
    }

    uint32_t sUint32Column::Size() { return mData.size(); }

    void sUint32Column::Render( int aRow, ImVec2 aSize )
    {
        std::string lText     = fmt::format( "{}", mData[aRow] );
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

    void sUint32Column::Clear()
    {
        sTableColumn::Clear();
        mData.clear();
    }

    void *sUint32Column::UIUint32Column_Create()
    {
        auto lNewColumn = new sUint32Column();

        return static_cast<void *>( lNewColumn );
    }

    void *sUint32Column::UIUint32Column_CreateFull( void *aHeader, float aInitialSize )
    {
        auto lHeader    = DotNetRuntime::NewString( static_cast<MonoString *>( aHeader ) );
        auto lNewColumn = new sUint32Column( lHeader, aInitialSize );

        return static_cast<void *>( lNewColumn );
    }

    void sUint32Column::UIUint32Column_Destroy( void *aSelf ) { delete static_cast<sUint32Column *>( aSelf ); }

    void sUint32Column::UIUint32Column_Clear( void *aSelf )
    {
        auto lSelf = static_cast<sUint32Column *>( aSelf );

        lSelf->Clear();
    }

    void sUint32Column::UIUint32Column_SetData( void *aSelf, void *aValue )
    {
        auto lSelf = static_cast<sUint32Column *>( aSelf );

        lSelf->mData = DotNetRuntime::AsVector<uint32_t>( static_cast<MonoObject *>( aValue ) );
    }

    void sUint32Column::UIUint32Column_SetDataWithForegroundColor( void *aSelf, void *aValue, void *aForegroundColor )
    {
        auto lSelf = static_cast<sUint32Column *>( aSelf );

        lSelf->mData = DotNetRuntime::AsVector<uint32_t>( static_cast<MonoObject *>( aValue ) );
        lSelf->mForegroundColor.clear();
        for( auto const &x : DotNetRuntime::AsVector<ImVec4>( static_cast<MonoObject *>( aForegroundColor ) ) )
            lSelf->mForegroundColor.push_back( ImColor( x ) );
    }

    void sUint32Column::UIUint32Column_SetDataWithForegroundAndBackgroundColor( void *aSelf, void *aValue, void *aForegroundColor,
                                                                                void *aBackroundColor )
    {
        auto lSelf = static_cast<sUint32Column *>( aSelf );

        lSelf->mData = DotNetRuntime::AsVector<uint32_t>( static_cast<MonoObject *>( aValue ) );

        lSelf->mForegroundColor.clear();
        for( auto const &x : DotNetRuntime::AsVector<ImVec4>( static_cast<MonoObject *>( aForegroundColor ) ) )
            lSelf->mForegroundColor.push_back( ImColor( x ) );

        lSelf->mBackgroundColor.clear();
        for( auto const &x : DotNetRuntime::AsVector<ImVec4>( static_cast<MonoObject *>( aBackroundColor ) ) )
            lSelf->mBackgroundColor.push_back( ImColor( x ) );
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

    void sStringColumn::Clear()
    {
        sTableColumn::Clear();
        mData.clear();
    }

    void *sStringColumn::UIStringColumn_Create()
    {
        auto lNewColumn = new sStringColumn();

        return static_cast<void *>( lNewColumn );
    }

    void *sStringColumn::UIStringColumn_CreateFull( void *aHeader, float aInitialSize )
    {
        auto lHeader    = DotNetRuntime::NewString( static_cast<MonoString *>( aHeader ) );
        auto lNewColumn = new sStringColumn( lHeader, aInitialSize );

        return static_cast<void *>( lNewColumn );
    }

    void sStringColumn::UIStringColumn_Destroy( void *aSelf ) { delete static_cast<sStringColumn *>( aSelf ); }

    void sStringColumn::UIStringColumn_Clear( void *aSelf )
    {
        auto lSelf = static_cast<sStringColumn *>( aSelf );

        lSelf->Clear();
    }

    void sStringColumn::UIStringColumn_SetData( void *aSelf, void *aValue )
    {
        auto lSelf = static_cast<sStringColumn *>( aSelf );

        lSelf->mData.clear();
        for( auto const &x : DotNetRuntime::AsVector<MonoString *>( static_cast<MonoObject *>( aValue ) ) )
            lSelf->mData.push_back( DotNetRuntime::NewString( x ) );
    }

    void sStringColumn::UIStringColumn_SetDataWithForegroundColor( void *aSelf, void *aValue, void *aForegroundColor )
    {
        auto lSelf = static_cast<sStringColumn *>( aSelf );

        lSelf->mData.clear();
        for( auto const &x : DotNetRuntime::AsVector<MonoString *>( static_cast<MonoObject *>( aValue ) ) )
            lSelf->mData.push_back( DotNetRuntime::NewString( x ) );

        lSelf->mForegroundColor.clear();
        for( auto const &x : DotNetRuntime::AsVector<ImVec4>( static_cast<MonoObject *>( aForegroundColor ) ) )
            lSelf->mForegroundColor.push_back( ImColor( x ) );
    }

    void sStringColumn::UIStringColumn_SetDataWithForegroundAndBackgroundColor( void *aSelf, void *aValue, void *aForegroundColor,
                                                                                void *aBackroundColor )
    {
        auto lSelf = static_cast<sStringColumn *>( aSelf );

        lSelf->mData.clear();
        for( auto const &x : DotNetRuntime::AsVector<MonoString *>( static_cast<MonoObject *>( aValue ) ) )
            lSelf->mData.push_back( DotNetRuntime::NewString( x ) );

        lSelf->mForegroundColor.clear();
        for( auto const &x : DotNetRuntime::AsVector<ImVec4>( static_cast<MonoObject *>( aForegroundColor ) ) )
            lSelf->mForegroundColor.push_back( ImColor( x ) );

        lSelf->mBackgroundColor.clear();
        for( auto const &x : DotNetRuntime::AsVector<ImVec4>( static_cast<MonoObject *>( aBackroundColor ) ) )
            lSelf->mBackgroundColor.push_back( ImColor( x ) );
    }

} // namespace SE::Core