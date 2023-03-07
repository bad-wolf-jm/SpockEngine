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

    UILinkElementTable::UILinkElementTable() : UITable()
    {
        SetRowHeight( 20.0f );

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
        AddColumn(mEventStatus);

        mReflectanceType = New<sFloat64Column>( "mReflectanceType", 75.0f, "{:.3f} dB/km", "\xe2\x88\x9e km" );
        AddColumn(mReflectanceType);
    }
} // namespace SE::OtdrEditor