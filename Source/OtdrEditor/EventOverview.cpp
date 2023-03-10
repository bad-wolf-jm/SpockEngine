#include "EventOverview.h"

namespace SE::OtdrEditor
{
    EventOverview::EventOverview()
        : UIBoxLayout( eBoxLayoutOrientation::VERTICAL )
    {

        const float lItemHeight = 20.0f;

        mEventOverview = New<UILabel>( "Event properties" );
        mEventOverview->SetAlignment( eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        Add( mEventOverview.get(), 30.0f, false, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        mEventOverviewLayout         = New<UIBoxLayout>( eBoxLayoutOrientation::HORIZONTAL );
        mEventOverviewPropertyLayout = New<UIBoxLayout>( eBoxLayoutOrientation::VERTICAL );
        mEventOverviewLayout->Add( nullptr, 25.0f, false, false );
        mEventOverviewLayout->Add( mEventOverviewPropertyLayout.get(), true, true );
        mWavelength = New<UIPropertyValue>( "Overview length:" );
        mEventOverviewPropertyLayout->Add( mWavelength.get(), lItemHeight, false, true );
        mDetectionTrace = New<UIPropertyValue>( "Link length:" );
        mEventOverviewPropertyLayout->Add( mDetectionTrace.get(), lItemHeight, false, true );
        mPositionTolerance = New<UIPropertyValue>( "Completion:" );
        mEventOverviewPropertyLayout->Add( mPositionTolerance.get(), lItemHeight, false, true );
        mCurveLevel = New<UIPropertyValue>( "Overview length:" );
        mEventOverviewPropertyLayout->Add( mCurveLevel.get(), lItemHeight, false, true );
        mEndOrNoiseLevel = New<UIPropertyValue>( "Link length:" );
        mEventOverviewPropertyLayout->Add( mEndOrNoiseLevel.get(), lItemHeight, false, true );
        mEstimatedCurveLevel = New<UIPropertyValue>( "Completion:" );
        mEventOverviewPropertyLayout->Add( mEstimatedCurveLevel.get(), lItemHeight, false, true );
        mEstimatedEndLevel = New<UIPropertyValue>( "Overview length:" );
        mEventOverviewPropertyLayout->Add( mEstimatedEndLevel.get(), lItemHeight, false, true );
        mEstimatedLoss = New<UIPropertyValue>( "Link length:" );
        mEventOverviewPropertyLayout->Add( mEstimatedLoss.get(), lItemHeight, false, true );
        mPeakSNR = New<UIPropertyValue>( "Completion:" );
        mEventOverviewPropertyLayout->Add( mPeakSNR.get(), lItemHeight, false, true );
        mPeakTrace = New<UIPropertyValue>( "Overview length:" );
        mEventOverviewPropertyLayout->Add( mPeakTrace.get(), lItemHeight, false, true );
        mPeakPosition = New<UIPropertyValue>( "Link length:" );
        mEventOverviewPropertyLayout->Add( mPeakPosition.get(), lItemHeight, false, true );
        mPeakPower = New<UIPropertyValue>( "Completion:" );
        mEventOverviewPropertyLayout->Add( mPeakPower.get(), lItemHeight, false, true );
        mPeakSaturation = New<UIPropertyValue>( "Overview length:" );
        mEventOverviewPropertyLayout->Add( mPeakSaturation.get(), lItemHeight, false, true );
        mFresnelCorrection = New<UIPropertyValue>( "Link length:" );
        mEventOverviewPropertyLayout->Add( mFresnelCorrection.get(), lItemHeight, false, true );
        mUseSinglePulse = New<UIPropertyValue>( "Completion:" );
        mEventOverviewPropertyLayout->Add( mUseSinglePulse.get(), lItemHeight, false, true );
        Add( mEventOverviewLayout.get(), false, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );

        mPreviousSectionTitle = New<UILabel>( "Previous section" );
        mPreviousSectionTitle->SetAlignment( eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        Add( mPreviousSectionTitle.get(), 30.0f, false, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        mPreviousSectionLayout         = New<UIBoxLayout>( eBoxLayoutOrientation::HORIZONTAL );
        mPreviousSectionPropertyLayout = New<UIBoxLayout>( eBoxLayoutOrientation::VERTICAL );
        mPreviousSectionLayout->Add( nullptr, 25.0f, false, false );
        mPreviousSectionLayout->Add( mPreviousSectionPropertyLayout.get(), true, true );
        mPreviousRbsTrace = New<UIPropertyValue>( "Trace:" );
        mPreviousSectionPropertyLayout->Add( mPreviousRbsTrace.get(), lItemHeight, false, true );
        mPreviousRbsNoise = New<UIPropertyValue>( "Noise:" );
        mPreviousSectionPropertyLayout->Add( mPreviousRbsNoise.get(), lItemHeight, false, true );
        mPreviousRbsSaturation = New<UIPropertyValue>( "Saturation:" );
        mPreviousSectionPropertyLayout->Add( mPreviousRbsSaturation.get(), lItemHeight, false, true );
        mPreviousLsaRange = New<UIPropertyValue>( "LSA range:" );
        mPreviousSectionPropertyLayout->Add( mPreviousLsaRange.get(), lItemHeight, false, true );
        mPreviousLsaSlope = New<UIPropertyValue>( "Lsa slope:" );
        mPreviousSectionPropertyLayout->Add( mPreviousLsaSlope.get(), lItemHeight, false, true );
        mPreviousLsaOffset = New<UIPropertyValue>( "Lsa offset:" );
        mPreviousSectionPropertyLayout->Add( mPreviousLsaOffset.get(), lItemHeight, false, true );
        mPreviousLsaMean = New<UIPropertyValue>( "lsa mean:" );
        mPreviousSectionPropertyLayout->Add( mPreviousLsaMean.get(), lItemHeight, false, true );
        mPreviousLsaSlopeError = New<UIPropertyValue>( "Lsa slope error:" );
        mPreviousSectionPropertyLayout->Add( mPreviousLsaSlopeError.get(), lItemHeight, false, true );
        mPreviousLsaLinear = New<UIPropertyValue>( "LSA linear:" );
        mPreviousSectionPropertyLayout->Add( mPreviousLsaLinear.get(), lItemHeight, false, true );
        mPreviousLsaError = New<UIPropertyValue>( "LSA error:" );
        mPreviousSectionPropertyLayout->Add( mPreviousLsaError.get(), lItemHeight, false, true );
        mPreviousLsaCrossings = New<UIPropertyValue>( "LSA crossings:" );
        mPreviousSectionPropertyLayout->Add( mPreviousLsaCrossings.get(), lItemHeight, false, true );
        mPreviousLsaSections = New<UIPropertyValue>( "LSA sections:" );
        mPreviousSectionPropertyLayout->Add( mPreviousLsaSections.get(), lItemHeight, false, true );
        Add( mPreviousSectionLayout.get(), false, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );

        mNextSectionTitle = New<UILabel>( "Previous section" );
        mNextSectionTitle->SetAlignment( eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        Add( mNextSectionTitle.get(), 30.0f, false, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        mNextSectionLayout         = New<UIBoxLayout>( eBoxLayoutOrientation::HORIZONTAL );
        mNextSectionPropertyLayout = New<UIBoxLayout>( eBoxLayoutOrientation::VERTICAL );
        mNextSectionLayout->Add( nullptr, 25.0f, false, false );
        mNextSectionLayout->Add( mNextSectionPropertyLayout.get(), true, true );
        mNextRbsTrace = New<UIPropertyValue>( "Trace:" );
        mNextSectionPropertyLayout->Add( mPreviousRbsTrace.get(), lItemHeight, false, true );
        mNextRbsNoise = New<UIPropertyValue>( "Noise:" );
        mNextSectionPropertyLayout->Add( mPreviousRbsNoise.get(), lItemHeight, false, true );
        mNextRbsSaturation = New<UIPropertyValue>( "Saturation:" );
        mNextSectionPropertyLayout->Add( mPreviousRbsSaturation.get(), lItemHeight, false, true );
        mNextLsaRange = New<UIPropertyValue>( "LSA range:" );
        mNextSectionPropertyLayout->Add( mPreviousLsaRange.get(), lItemHeight, false, true );
        mNextLsaSlope = New<UIPropertyValue>( "Lsa slope:" );
        mNextSectionPropertyLayout->Add( mPreviousLsaSlope.get(), lItemHeight, false, true );
        mNextLsaOffset = New<UIPropertyValue>( "Lsa offset:" );
        mNextSectionPropertyLayout->Add( mPreviousLsaOffset.get(), lItemHeight, false, true );
        mNextLsaMean = New<UIPropertyValue>( "lsa mean:" );
        mNextSectionPropertyLayout->Add( mPreviousLsaMean.get(), lItemHeight, false, true );
        mNextLsaSlopeError = New<UIPropertyValue>( "Lsa slope error:" );
        mNextSectionPropertyLayout->Add( mPreviousLsaSlopeError.get(), lItemHeight, false, true );
        mNextLsaLinear = New<UIPropertyValue>( "LSA linear:" );
        mNextSectionPropertyLayout->Add( mPreviousLsaLinear.get(), lItemHeight, false, true );
        mNextLsaError = New<UIPropertyValue>( "LSA error:" );
        mNextSectionPropertyLayout->Add( mPreviousLsaError.get(), lItemHeight, false, true );
        mNextLsaCrossings = New<UIPropertyValue>( "LSA crossings:" );
        mNextSectionPropertyLayout->Add( mPreviousLsaCrossings.get(), lItemHeight, false, true );
        mNextLsaSections = New<UIPropertyValue>( "LSA sections:" );
        mNextSectionPropertyLayout->Add( mPreviousLsaSections.get(), lItemHeight, false, true );
        Add( mNextSectionLayout.get(), true, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
    }

    void EventOverview::SetData( Ref<MonoScriptInstance> aEventOverview )
    {
        // Overview

        // Previous section

        // Next section
    }
} // namespace SE::OtdrEditor