#include "TestFailResultTable.h"

#include <cmath>
#include <fmt/core.h>
#include <fstream>
#include <iostream>
#include <locale>
#include <numeric>

#include "Core/Profiling/BlockTimer.h"
#include "Mono/MonoRuntime.h"

namespace SE::OtdrEditor
{
    using namespace SE::Core;

    UITestFailResultTable::UITestFailResultTable()
        : UITable()
    {
        SetRowHeight( 20.0f );

        mFilename = New<sStringColumn>( "mFilename", 75.0f );
        AddColumn( mFilename );

        mMessage = New<sStringColumn>( "mMessage", 75.0f );
        AddColumn( mMessage );

        mPhysicalEventPosition = New<sStringColumn>( "mPhysicalEventPosition", 75.0f );
        AddColumn( mPhysicalEventPosition );

        mLinkElementIndex = New<sStringColumn>( "mLinkElementIndex", 75.0f );
        AddColumn( mLinkElementIndex );

        mSubLinkElementIndex = New<sStringColumn>( "mSubLinkElementIndex", 75.0f );
        AddColumn( mSubLinkElementIndex );

        mPhysicalEventIndex = New<sStringColumn>( "mPhysicalEventIndex", 75.0f );
        AddColumn( mPhysicalEventIndex );

        mLinkElementPosition = New<sStringColumn>( "mLinkElementPosition", 75.0f );
        AddColumn( mLinkElementPosition );

        mIsSubElement = New<sStringColumn>( "mIsSubElement", 75.0f );
        AddColumn( mIsSubElement );

        mWavelength = New<sStringColumn>( "mWavelength", 75.0f );
        AddColumn( mWavelength );


        mSinglePulseTraceIndex = New<sStringColumn>( "mSinglePulseTraceIndex", 75.0f );
        AddColumn( mSinglePulseTraceIndex );

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
            mFilename->mData.push_back( lDataRow.mFilename );
            mLinkElementIndex->mData.push_back( lDataRow.mLinkElementIndex );
            mSubLinkElementIndex->mData.push_back( lDataRow.mSubLinkElementIndex );
            mPhysicalEventIndex->mData.push_back( lDataRow.mPhysicalEventIndex );
            mLinkElementPosition->mData.push_back( lDataRow.mLinkElementPosition );
            mIsSubElement->mData.push_back( lDataRow.mIsSubElement );
            mWavelength->mData.push_back( lDataRow.mWavelength );
            mPhysicalEventPosition->mData.push_back( lDataRow.mPhysicalEventPosition );
            mSinglePulseTraceIndex->mData.push_back( lDataRow.mSinglePulseTraceIndex );
            mMessage->mData.push_back( lDataRow.mMessage );
        }
    }

    void UITestFailResultTable::Clear()
    {
        mRowBackgroundColor.clear();
        mFilename->mData.clear();
        mLinkElementIndex->mData.clear();
        mSubLinkElementIndex->mData.clear();
        mPhysicalEventIndex->mData.clear();
        mLinkElementPosition->mData.clear();
        mIsSubElement->mData.clear();
        mWavelength->mData.clear();
        mPhysicalEventPosition->mData.clear();
        mSinglePulseTraceIndex->mData.clear();
        mMessage->mData.clear();
    }

    // std::vector<sLinkElement> UITestFailResultTable::GetElementsByIndex( uint32_t aElementIndex )
    // {
    //     std::vector<sLinkElement> lResult;

    //     for( auto const &x : mEventDataVector )
    //         if( x.mLinkIndex == aElementIndex ) lResult.push_back( x );

    //     return std::move( lResult );
    // }
} // namespace SE::OtdrEditor