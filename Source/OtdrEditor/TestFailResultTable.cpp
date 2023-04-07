#include "TestFailResultTable.h"

#include <cmath>
#include <fmt/core.h>
#include <fstream>
#include <iostream>
#include <locale>
#include <numeric>

#include "Core/Profiling/BlockTimer.h"
#include "DotNet/Runtime.h"

namespace SE::OtdrEditor
{
    using namespace SE::Core;

    UITestFailResultTable::UITestFailResultTable()
        : UITable()
    {
        SetRowHeight( 20.0f );

        mTestDate = New<sStringColumn>( "Date", 125.0f );
        AddColumn( mTestDate );

        mTestName = New<sStringColumn>( "Test", 175.0f );
        AddColumn( mTestName );
 
        mMessage = New<sStringColumn>( "Message", 75.0f );
        AddColumn( mMessage );

        mLinkElementPosition = New<sFloat64Column>( "Element Position", 75.0f, "{:.4f} km", "" );
        AddColumn( mLinkElementPosition );

        mPhysicalEventPosition = New<sFloat64Column>( "Physical Event Position", 75.0f, "{:.4f} km", "");
        AddColumn( mPhysicalEventPosition );

        mWavelength = New<sFloat64Column>( "mWavelength", 75.0f, "{:.1f} nm", "" );
        AddColumn( mWavelength );

        mFilename = New<sStringColumn>( "File", 775.0f );
        AddColumn( mFilename );

        UITable::OnRowClicked(
            [&]( uint32_t aRow )
            {
                if( mOnElementClicked ) mOnElementClicked( mEventDataVector[aRow] );
            } );
    }

    void UITestFailResultTable::OnElementClicked( std::function<void( sTestFailElement const & )> const &aOnRowClicked )
    {
        mOnElementClicked = aOnRowClicked;
    }

    void UITestFailResultTable::SetData( std::vector<sTestFailElement> const &aData )
    {
        Clear();

        mEventDataVector = std::move( aData );

        for( auto const &lDataRow : mEventDataVector )
        {
            mTestName->mData.push_back( lDataRow.mTestName );
            mTestDate->mData.push_back( lDataRow.mTestDate );
            mFilename->mData.push_back( fs::path(lDataRow.mFilename).string() );
            mLinkElementPosition->mData.push_back( lDataRow.mLinkElementPosition * 0.001f );
            mWavelength->mData.push_back( lDataRow.mWavelength * 1e9 );
            mPhysicalEventPosition->mData.push_back( lDataRow.mPhysicalEventPosition * 0.001f );
            mMessage->mData.push_back( lDataRow.mMessage );
        }
    }

    void UITestFailResultTable::Clear()
    {
        mRowBackgroundColor.clear();
        mTestName->mData.clear();
        mTestDate->mData.clear();
        mFilename->mData.clear();
        mLinkElementPosition->mData.clear();
        mWavelength->mData.clear();
        mPhysicalEventPosition->mData.clear();
        mMessage->mData.clear();
    }
} // namespace SE::OtdrEditor