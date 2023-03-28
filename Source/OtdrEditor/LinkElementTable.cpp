#include "LinkElementTable.h"

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

    UILinkElementTable::UILinkElementTable()
        : UITable()
    {
        SetRowHeight( 20.0f );

        mIndex = New<sStringColumn>( "ID", 25.0f );
        AddColumn( mIndex );

        mDiagnosicCount = New<sStringColumn>( ICON_FA_COMMENTS, 15.0f );
        AddColumn( mDiagnosicCount );

        mPositionColumn = New<sFloat64Column>( "Position", 75.0f, "{:.4f} km", "N.a.N." );
        AddColumn( mPositionColumn );

        mWavelength = New<sFloat64Column>( "Wavelength", 75.0f, "{:.1f} nm", "N.a.N." );
        AddColumn( mWavelength );

        mType = New<sStringColumn>( "Type", 75.0f );
        AddColumn( mType );

        mEventType = New<sStringColumn>( "mEventType", 75.0f );
        AddColumn( mEventType );

        mStatus = New<sStringColumn>( "Status", 75.0f );
        AddColumn( mStatus );

        mLoss = New<sFloat64Column>( "Loss", 75.0f, "{:.3f} dB", "N.a.N." );
        AddColumn( mLoss );

        mReflectance = New<sFloat64Column>( "Reflectance", 75.0f, "{:.2f} dB", "N.a.N." );
        AddColumn( mReflectance );

        mPeakPower = New<sFloat64Column>( "Peak power", 75.0f, "{:.2f} dB", "N.a.N." );
        AddColumn( mPeakPower );

        UITable::OnRowClicked(
            [&]( uint32_t aRow )
            {
                if( mOnElementClicked ) mOnElementClicked( mEventDataVector[aRow] );
            } );
    }

    void UILinkElementTable::OnElementClicked( std::function<void( sLinkElement const & )> const &aOnRowClicked )
    {
        mOnElementClicked = aOnRowClicked;
    }

    void UILinkElementTable::SetData( std::vector<sLinkElement> const &aData )
    {
        Clear();

        mEventDataVector = std::move( aData );

        auto StringJoin = []( std::vector<std::string> aList )
        {
            return aList.empty() ? "N/A"
                                 : std::accumulate( aList.begin(), aList.end(), std::string(),
                                                    []( const std::string &a, const std::string &b ) -> std::string
                                                    { return a + ( a.length() > 0 ? ", " : "" ) + b; } );
        };

        for( auto const &lE : mEventDataVector )
        {
            auto &lLinkElement   = *lE.mLinkElement;
            auto &lPhysicalEvent = *lE.mPhysicalEvent;
            auto &lAttributes    = *lE.mAttributes;

            if( lE.mLinkIndex % 2 )
                mRowBackgroundColor.push_back( IM_COL32( 2, 2, 2, 255 ) );
            else
                mRowBackgroundColor.push_back( IM_COL32( 9, 9, 9, 255 ) );

            auto lHidden    = lLinkElement.GetPropertyValue<bool>( "Hidden" );
            auto lTextColor = lHidden ? ImGui::GetStyleColorVec4( ImGuiCol_TextDisabled ) : ImGui::GetStyleColorVec4( ImGuiCol_Text );
            auto lOtdrPhysicalEvent = lPhysicalEvent.GetPropertyValue( "PhysicalEvent", "Metrino.Otdr.PhysicalEvent" );

            if( lE.mIsSubElement )
                mIndex->mData.push_back( fmt::format( " {} {}", ICON_FA_ANGLE_RIGHT, lE.mSubLinkIndex ) );
            else
                mIndex->mData.push_back( fmt::format( "{}", lE.mLinkIndex ) );
            mIndex->mForegroundColor.push_back( (uint32_t)ImColor( lTextColor ) );

            std::string lSplitterSource = lLinkElement.GetPropertyValue<bool>( "TwoByNSplitter" ) ? "2" : "1";
            if( lLinkElement.GetPropertyValue<eLinkElementType>( "Type" ) == eLinkElementType::Splitter )
            {
                auto lSplitterConfiguration =
                    fmt::format( "{} ({}\xC3\x97{})", ToString( lLinkElement.GetPropertyValue<eLinkElementType>( "Type" ) ),
                                 lSplitterSource, lLinkElement.GetPropertyValue<int>( "SplitterRatio" ) );
                mType->mData.push_back( lSplitterConfiguration );
            }
            else
            {
                mType->mData.push_back( ToString( lLinkElement.GetPropertyValue<eLinkElementType>( "Type" ) ) );
            }
            mType->mForegroundColor.push_back( (uint32_t)ImColor( lTextColor ) );

            mStatus->mData.push_back( StringJoin( LinkStatusToString( lLinkElement.GetPropertyValue<int>( "Status" ) ) ) );
            mStatus->mForegroundColor.push_back( (uint32_t)ImColor( lTextColor ) );

            if( lE.mDiagnosicCount == 1 )
                mDiagnosicCount->mData.push_back( ICON_FA_COMMENT );
            else if( lE.mDiagnosicCount > 1 )
                mDiagnosicCount->mData.push_back( ICON_FA_COMMENTS );
            else
                mDiagnosicCount->mData.push_back( ICON_FA_COMMENT_O );
            mDiagnosicCount->mForegroundColor.push_back( (uint32_t)ImColor( lTextColor ) );

            mWavelength->mData.push_back( lPhysicalEvent.GetPropertyValue<double>( "Wavelength" ) * 1e9 );
            mWavelength->mForegroundColor.push_back( (uint32_t)ImColor( lTextColor ) );

            mPositionColumn->mData.push_back( lLinkElement.GetPropertyValue<double>( "Position" ) * 0.001f );
            mPositionColumn->mForegroundColor.push_back( (uint32_t)ImColor( lTextColor ) );

            mLoss->mData.push_back( lOtdrPhysicalEvent->GetPropertyValue<double>( "Loss" ) );

            if( lE.mLossPassFail == ePassFail::PASS )
                mLoss->mForegroundColor.emplace_back( IM_COL32( 0, 255, 0, lTextColor.w * 255 ) );
            else if( lE.mLossPassFail == ePassFail::FAIL )
                mLoss->mForegroundColor.emplace_back( IM_COL32( 255, 0, 0, lTextColor.w * 255 ) );
            else
                mLoss->mForegroundColor.emplace_back( (uint32_t)ImColor( lTextColor ) );

            mReflectance->mData.push_back( lOtdrPhysicalEvent->GetPropertyValue<double>( "Reflectance" ) );
            if( lE.mReflectancePassFail == ePassFail::PASS )
                mReflectance->mForegroundColor.emplace_back( IM_COL32( 0, 255, 0, lTextColor.w * 255 ) );
            else if( lE.mReflectancePassFail == ePassFail::FAIL )
                mReflectance->mForegroundColor.emplace_back( IM_COL32( 255, 0, 0, lTextColor.w * 255 ) );
            else
                mReflectance->mForegroundColor.emplace_back( (uint32_t)ImColor( lTextColor ) );

            mEventType->mData.push_back( ToString( lOtdrPhysicalEvent->GetPropertyValue<eEventType>( "Type" ) ) );
            mEventType->mForegroundColor.push_back( (uint32_t)ImColor( lTextColor ) );

            if( lAttributes )
                mPeakPower->mData.push_back( lAttributes.GetPropertyValue<double>( "PeakPower" ) );
            else
                mPeakPower->mData.push_back( nan( "" ) );

            mPeakPower->mForegroundColor.push_back( (uint32_t)ImColor( lTextColor ) );
        }
    }

    void UILinkElementTable::Clear()
    {
        mRowBackgroundColor.clear();

        mIndex->Clear();
        mType->Clear();
        mStatus->Clear();
        mDiagnosicCount->Clear();
        mWavelength->Clear();
        mPositionColumn->Clear();
        mLoss->Clear();
        mReflectance->Clear();
        mPeakPower->Clear();
        mEventType->Clear();
    }

    std::vector<sLinkElement> UILinkElementTable::GetElementsByIndex( uint32_t aElementIndex )
    {
        std::vector<sLinkElement> lResult;

        for( auto const &x : mEventDataVector )
            if( x.mLinkIndex == aElementIndex ) lResult.push_back( x );

        return std::move( lResult );
    }
} // namespace SE::OtdrEditor