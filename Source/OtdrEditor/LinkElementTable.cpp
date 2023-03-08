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

        mType = New<sStringColumn>( "Type", 75.0f );
        AddColumn( mType );

        mStatus = New<sFloat64Column>( "Status", 75.0f, "{:.3f}", "N.a.N." );

        AddColumn( mStatus );
        mPositionColumn = New<sFloat64Column>( "Position", 75.0f, "{:.3f} km", "N.a.N." );
        AddColumn( mPositionColumn );

        mDiagnosicCount = New<sFloat64Column>( "Diagnostic", 75.0f, "{:.3f}",  "N.a.N." );
        AddColumn( mDiagnosicCount );

        mPreviousFiberSectionA = New<sFloat64Column>( "mPreviousFiberSectionA", 75.0f, "{:.3f} km", "N.a.N." );
        AddColumn( mPreviousFiberSectionA );

        mPreviousFiberSectionB = New<sFloat64Column>( "mPreviousFiberSectionB", 75.0f, "{:.3f} km", "N.a.N." );
        AddColumn( mPreviousFiberSectionB );

        mPreviousFiberSectionLength = New<sFloat64Column>( "mPreviousFiberSectionLength", 75.0f, "{:.3f} km", "N.a.N." );
        AddColumn( mPreviousFiberSectionLength );

        mPreviousFiberSectionLoss = New<sFloat64Column>( "mPreviousFiberSectionLoss", 75.0f, "{:.3f} km", "N.a.N." );
        AddColumn( mPreviousFiberSectionLoss );

        mPreviousFiberSectionAttenuation =
            New<sFloat64Column>( "mPreviousFiberSectionAttenuation", 75.0f, "{:.3f} dB/km", "N.a.N." );
        AddColumn( mPreviousFiberSectionAttenuation );

        mLoss = New<sFloat64Column>( "Loss", 75.0f, "{:.3f} dB", "N.a.N." );
        AddColumn( mLoss );

        mLossPassFail = New<sFloat64Column>( "LossPF", 75.0f, "{:.3f}", "N.a.N." );
        AddColumn( mLossPassFail );

        mLossError = New<sFloat64Column>( "Loss Error", 75.0f, "{:.3f} dB", "N.a.N." );
        AddColumn( mLossError );

        mReflectance = New<sFloat64Column>( "Reflectance", 75.0f, "{:.3f} dB", "N.a.N." );
        AddColumn( mReflectance );

        mReflectancePassFail = New<sFloat64Column>( "ReflectancePF", 75.0f, "{:.3f}", "N.a.N." );
        AddColumn( mReflectancePassFail );

        mCurveLevelColumn = New<sFloat64Column>( "Level", 75.0f, "{:.3f} dB", "N.a.N." );
        AddColumn( mCurveLevelColumn );

        mPositionTolerance = New<sFloat64Column>( "mPositionTolerance", 75.0f, "{:.3f} km", "N.a.N.m" );
        AddColumn( mPositionTolerance );

        mEventType = New<sStringColumn>( "mEventType", 75.0f );
        AddColumn( mEventType );

        mEventStatus = New<sFloat64Column>( "mEventStatus", 75.0f, "{:.3f}", "N.a.N." );
        AddColumn( mEventStatus );

        mReflectanceType = New<sStringColumn>( "mReflectanceType", 75.0f );
        AddColumn( mReflectanceType );
    }

    void UILinkElementTable::SetData( std::vector<sLinkElement> const &aData )
    {
        mEventDataVector = std::move( aData );

        mType->mData.clear();
        for( auto const &lE : mEventDataVector ) mType->mData.push_back( ToString(lE.mType) );

        mStatus->mData.clear();
        for( auto const &lE : mEventDataVector ) mStatus->mData.push_back( 0.0f );

        mDiagnosicCount->mData.clear();
        for( auto const &lE : mEventDataVector ) mDiagnosicCount->mData.push_back( lE.mDiagnosicCount );

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
        for( auto const &lE : mEventDataVector ) mEventType->mData.push_back( ToString(lE.mEventType) );

        mEventStatus->mData.clear();
        for( auto const &lE : mEventDataVector ) mEventStatus->mData.push_back( 0.0f );

        mReflectanceType->mData.clear();
        for( auto const &lE : mEventDataVector ) mReflectanceType->mData.push_back(ToString(lE.mReflectanceType) );
    }

} // namespace SE::OtdrEditor