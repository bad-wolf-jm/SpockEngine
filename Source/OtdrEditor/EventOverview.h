#pragma once

#include "Core/Memory.h"

#include "UI/Components/Label.h"
#include "UI/Components/PropertyValue.h"
#include "UI/Layouts/BoxLayout.h"

#include "DotNet/DotNetInstance.h"

namespace SE::OtdrEditor
{
    using namespace SE::Core;

    class SectionOverview : public UIBoxLayout
    {
      public:
        SectionOverview(std::string const& aTitle);
        ~SectionOverview() = default;

        void SetData(Ref<DotNetInstance> aRbsData);

      protected:
        Ref<UILabel>         mSectionTitle;
        Ref<UIBoxLayout>     mSectionLayout;
        Ref<UIBoxLayout>     mSectionPropertyLayout;
        Ref<UIPropertyValue> mRbsTrace;
        Ref<UIPropertyValue> mRbsNoise;
        Ref<UIPropertyValue> mRbsSaturation;
        Ref<UIPropertyValue> mLsaRange;
        Ref<UIPropertyValue> mLsaSlope;
        Ref<UIPropertyValue> mLsaOffset;
        Ref<UIPropertyValue> mLsaMean;
        Ref<UIPropertyValue> mLsaSlopeError;
        Ref<UIPropertyValue> mLsaLinear;
        Ref<UIPropertyValue> mLsaError;
        Ref<UIPropertyValue> mLsaCrossings;
        Ref<UIPropertyValue> mLsaSections;
    };


    class EventOverview : public UIBoxLayout
    {
      public:
        EventOverview();
        ~EventOverview() = default;

        void SetData( Ref<DotNetInstance> aEventOverview, Ref<DotNetInstance> aAttributes );

      protected:
        Ref<UILabel>         mEventOverview;
        Ref<UIBoxLayout>     mEventOverviewLayout;
        Ref<UIBoxLayout>     mEventOverviewPropertyLayout;
        Ref<UIPropertyValue> mWavelength;
        Ref<UIPropertyValue> mDetectionTrace;
        Ref<UIPropertyValue> mPositionTolerance;
        Ref<UIPropertyValue> mCurveLevel;
        Ref<UIPropertyValue> mEndOrNoiseLevel;
        Ref<UIPropertyValue> mEstimatedCurveLevel;
        Ref<UIPropertyValue> mEstimatedEndLevel;
        Ref<UIPropertyValue> mEstimatedLoss;
        Ref<UIPropertyValue> mPeakSNR;
        Ref<UIPropertyValue> mPeakTrace;
        Ref<UIPropertyValue> mPeakPosition;
        Ref<UIPropertyValue> mPeakPower;
        Ref<UIPropertyValue> mPeakSaturation;
        Ref<UIPropertyValue> mFresnelCorrection;
        Ref<UIPropertyValue> mUseSinglePulse;

        Ref<SectionOverview> mPreviousSectionLayout;
        Ref<SectionOverview> mNextSectionLayout;
    };
} // namespace SE::OtdrEditor