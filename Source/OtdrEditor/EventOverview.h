#pragma once

#include "Core/Memory.h"

#include "UI/Components/Label.h"
#include "UI/Components/PropertyValue.h"
#include "UI/Layouts/BoxLayout.h"

#include "Mono/MonoScriptInstance.h"

namespace SE::OtdrEditor
{
    using namespace SE::Core;

    class EventOverview : public UIBoxLayout
    {
      public:
        EventOverview();
        ~EventOverview() = default;

        void SetData(Ref<MonoScriptInstance> aEventOverview);

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

        Ref<UILabel>         mPreviousSectionTitle;
        Ref<UIBoxLayout>     mPreviousSectionLayout;
        Ref<UIBoxLayout>     mPreviousSectionPropertyLayout;
        Ref<UIPropertyValue> mPreviousRbsTrace;
        Ref<UIPropertyValue> mPreviousRbsNoise;
        Ref<UIPropertyValue> mPreviousRbsSaturation;
        Ref<UIPropertyValue> mPreviousLsaRange;
        Ref<UIPropertyValue> mPreviousLsaSlope;
        Ref<UIPropertyValue> mPreviousLsaOffset;
        Ref<UIPropertyValue> mPreviousLsaMean;
        Ref<UIPropertyValue> mPreviousLsaSlopeError;
        Ref<UIPropertyValue> mPreviousLsaLinear;
        Ref<UIPropertyValue> mPreviousLsaError;
        Ref<UIPropertyValue> mPreviousLsaCrossings;
        Ref<UIPropertyValue> mPreviousLsaSections;

        Ref<UILabel>         mNextSectionTitle;
        Ref<UIBoxLayout>     mNextSectionLayout;
        Ref<UIBoxLayout>     mNextSectionPropertyLayout;
        Ref<UIPropertyValue> mNextRbsTrace;
        Ref<UIPropertyValue> mNextRbsNoise;
        Ref<UIPropertyValue> mNextRbsSaturation;
        Ref<UIPropertyValue> mNextLsaRange;
        Ref<UIPropertyValue> mNextLsaSlope;
        Ref<UIPropertyValue> mNextLsaOffset;
        Ref<UIPropertyValue> mNextLsaMean;
        Ref<UIPropertyValue> mNextLsaSlopeError;
        Ref<UIPropertyValue> mNextLsaLinear;
        Ref<UIPropertyValue> mNextLsaError;
        Ref<UIPropertyValue> mNextLsaCrossings;
        Ref<UIPropertyValue> mNextLsaSections;
    };
} // namespace SE::OtdrEditor