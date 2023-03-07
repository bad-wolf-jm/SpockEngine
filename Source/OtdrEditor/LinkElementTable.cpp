#include "LinkElementTable.h"

#include <cmath>
#include <fmt/core.h>
#include <fstream>
#include <iostream>
#include <locale>

#include "Core/Profiling/BlockTimer.h"

namespace SE::OtdrEditor
{

    using namespace SE::Core;
    // using namespace SE::Core::EntityComponentSystem::Components;

    UILinkElementTable::UILinkElementTable()
        : UITable()
    {
        SetRowHeight( 20.0f );

        mType = New<sFloat64Column>( "Position", 75.0f, "{:.3f} km", "\xe2\x88\x9e km" );
        AddColumn( mType );

        mPositionColumn = New<sFloat64Column>( "Position", 75.0f, "{:.3f} km", "\xe2\x88\x9e km" );
        AddColumn( mPositionColumn );

        mStatus = New<sFloat64Column>( "Status", 75.0f, "{:.3f}", "NaN" );
        AddColumn( mStatus );

        mDiagnosicCount = New<sFloat64Column>( "Diagnostic", 75.0f, "{:.3f}", "NaN" );
        AddColumn( mDiagnosicCount );

        mPreviousFiberSectionA = New<sFloat64Column>( "mPreviousFiberSectionA", 75.0f, "{:.3f} km", "\xe2\x88\x9e km" );
        AddColumn( mPreviousFiberSectionA );

        mPreviousFiberSectionB = New<sFloat64Column>( "mPreviousFiberSectionB", 75.0f, "{:.3f} km", "\xe2\x88\x9e km" );
        AddColumn( mPreviousFiberSectionB );

        mPreviousFiberSectionLength = New<sFloat64Column>( "mPreviousFiberSectionLength", 75.0f, "{:.3f} km", "\xe2\x88\x9e km" );
        AddColumn( mPreviousFiberSectionLength );

        mPreviousFiberSectionLoss = New<sFloat64Column>( "mPreviousFiberSectionLoss", 75.0f, "{:.3f} km", "\xe2\x88\x9e km" );
        AddColumn( mPreviousFiberSectionLoss );

        mPreviousFiberSectionAttenuation =
            New<sFloat64Column>( "mPreviousFiberSectionAttenuation", 75.0f, "{:.3f} dB/km", "\xe2\x88\x9e km" );
        AddColumn( mPreviousFiberSectionAttenuation );

        mLoss = New<sFloat64Column>( "Loss", 75.0f, "{:.3f} dB/km", "\xe2\x88\x9e km" );
        AddColumn( mLoss );

        mLossPassFail = New<sFloat64Column>( "LossPF", 75.0f, "{:.3f} dB/km", "\xe2\x88\x9e km" );
        AddColumn( mLossPassFail );

        mLossError = New<sFloat64Column>( "LossPF", 75.0f, "{:.3f} dB/km", "\xe2\x88\x9e km" );
        AddColumn( mLossError );

        mReflectance = New<sFloat64Column>( "Reflectance", 75.0f, "{:.3f} dB/km", "\xe2\x88\x9e km" );
        AddColumn( mReflectance );

        mReflectancePassFail = New<sFloat64Column>( "Reflectance", 75.0f, "{:.3f} dB/km", "\xe2\x88\x9e km" );
        AddColumn( mReflectancePassFail );

        mCurveLevelColumn = New<sFloat64Column>( "Level", 75.0f, "{:.3f} dB/km", "\xe2\x88\x9e km" );
        AddColumn( mCurveLevelColumn );

        mPositionTolerance = New<sFloat64Column>( "mPositionTolerance", 75.0f, "{:.3f} dB/km", "\xe2\x88\x9e km" );
        AddColumn( mPositionTolerance );

        mEventType = New<sFloat64Column>( "mEventType", 75.0f, "{:.3f} dB/km", "\xe2\x88\x9e km" );
        AddColumn( mEventType );

        mEventStatus = New<sFloat64Column>( "mEventStatus", 75.0f, "{:.3f} dB/km", "\xe2\x88\x9e km" );
        AddColumn( mEventStatus );

        mReflectanceType = New<sFloat64Column>( "mReflectanceType", 75.0f, "{:.3f} dB/km", "\xe2\x88\x9e km" );
        AddColumn( mReflectanceType );
    }

    void UILinkElementTable::SetData( std::vector<sLinkElement> const &aData )
    {
        mEventDataVector = std::move( aData );

        mType->mData.clear();
        for( auto const &lE : mEventDataVector ) mType->mData.push_back( 0.0f );

        mStatus->mData.clear();
        for( auto const &lE : mEventDataVector ) mStatus->mData.push_back( 0.0f );

        mDiagnosicCount->mData.clear();
        for( auto const &lE : mEventDataVector ) mDiagnosicCount->mData.push_back( 0.0f );

        mPositionColumn->mData.clear();
        for( auto const &lE : mEventDataVector ) mPositionColumn->mData.push_back( lE.mPosition * 0.001f );

        mPreviousFiberSectionA->mData.clear();
        for( auto const &lE : mEventDataVector ) mPreviousFiberSectionA->mData.push_back( lE.mPreviousFiberSectionA * 0.001f );

        mPreviousFiberSectionB->mData.clear();
        for( auto const &lE : mEventDataVector ) mPreviousFiberSectionB->mData.push_back( lE.mPreviousFiberSectionB * 0.001f );

        mPreviousFiberSectionLength->mData.clear();
        for( auto const &lE : mEventDataVector )
            mPreviousFiberSectionLength->mData.push_back( lE.mPreviousFiberSectionLength * 0.001f );

        mPreviousFiberSectionLoss->mData.clear();
        for( auto const &lE : mEventDataVector ) mPreviousFiberSectionLoss->mData.push_back( lE.mPreviousFiberSectionLoss * 0.001f );

        mPreviousFiberSectionAttenuation->mData.clear();
        for( auto const &lE : mEventDataVector )
            mPreviousFiberSectionAttenuation->mData.push_back( lE.mPreviousFiberSectionAttenuation * 0.001f );

        mLoss->mData.clear();
        for( auto const &lE : mEventDataVector ) mLoss->mData.push_back( lE.mLoss * 0.001f );

        mLossPassFail->mData.clear();
        for( auto const &lE : mEventDataVector ) mLossPassFail->mData.push_back( 0.0f );

        mLossError->mData.clear();
        for( auto const &lE : mEventDataVector ) mLossError->mData.push_back( 0.0f );

        mReflectance->mData.clear();
        for( auto const &lE : mEventDataVector ) mReflectance->mData.push_back( lE.mReflectance );

        mReflectancePassFail->mData.clear();
        for( auto const &lE : mEventDataVector ) mReflectancePassFail->mData.push_back( 0.0f );

        mCurveLevelColumn->mData.clear();
        for( auto const &lE : mEventDataVector ) mCurveLevelColumn->mData.push_back( lE.mCurveLevel );

        mPositionTolerance->mData.clear();
        for( auto const &lE : mEventDataVector ) mPositionTolerance->mData.push_back( lE.mPositionTolerance );

        mEventType->mData.clear();
        for( auto const &lE : mEventDataVector ) mEventType->mData.push_back( 0.0f );

        mEventStatus->mData.clear();
        for( auto const &lE : mEventDataVector ) mEventStatus->mData.push_back( 0.0f );

        mReflectanceType->mData.clear();
        for( auto const &lE : mEventDataVector ) mReflectanceType->mData.push_back( 0.0f );
    }

} // namespace SE::OtdrEditor