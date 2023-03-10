#pragma once

#include "Core/Memory.h"

#include "UI/Components/Label.h"
#include "UI/Components/PropertyValue.h"
#include "UI/Layouts/BoxLayout.h"

#include "Mono/MonoScriptInstance.h"

namespace SE::OtdrEditor
{
    using namespace SE::Core;

    class AcquisitionData : public UIBoxLayout
    {
      public:
        AcquisitionData();
        ~AcquisitionData() = default;

        void SetData(Ref<MonoScriptInstance> aRbsData);

      protected:
        Ref<UILabel>         mSectionTitle;
        Ref<UIBoxLayout>     mSectionLayout;
        Ref<UIBoxLayout>     mSectionPropertyLayout;
        Ref<UIPropertyValue> mWavelength;
        Ref<UIPropertyValue> mPulse;
        Ref<UIPropertyValue> mRange;
        Ref<UIPropertyValue> mStitchRange;
        Ref<UIPropertyValue> mLocalError;
        Ref<UIPropertyValue> mAverages;
        Ref<UIPropertyValue> mDecimationsPhases;
        Ref<UIPropertyValue> mDeltaT;
        Ref<UIPropertyValue> mTimeToOutput;
        Ref<UIPropertyValue> mTimeLASToOutput;
        Ref<UIPropertyValue> mTTOInternalSampling;
        Ref<UIPropertyValue> mInternalReflectance;
        Ref<UIPropertyValue> mSamplingDelay;
        Ref<UIPropertyValue> mRepetitionPeriod;
        Ref<UIPropertyValue> mAcquisitionTime;
        Ref<UIPropertyValue> mTzCode;
        Ref<UIPropertyValue> mBandwidth;
        Ref<UIPropertyValue> mAnalogGain;
        Ref<UIPropertyValue> mApdGain;
        Ref<UIPropertyValue> mStitchGain;
        Ref<UIPropertyValue> mFresnelCorrection;
        Ref<UIPropertyValue> mNormalizationFactor;
        Ref<UIPropertyValue> mHighBandwidthFilter;
        Ref<UIPropertyValue> mUnfilteredRmsNoise;
        Ref<UIPropertyValue> mLowBandwidthFilterNoise;
        Ref<UIPropertyValue> mHighBandwidthFiltedNoise;
        Ref<UIPropertyValue> mExpectedInjection;
        Ref<UIPropertyValue> mSaturationLevel;
        Ref<UIPropertyValue> mFresnelSaturation;
        Ref<UIPropertyValue> mSpeckeNoise;
        Ref<UIPropertyValue> mSaturationMaskRatio;
        Ref<UIPropertyValue> mApdID;
        Ref<UIPropertyValue> mRiseTime;
        Ref<UIPropertyValue> mFallTime;
        Ref<UIPropertyValue> mPositionTolerance;
        Ref<UIPropertyValue> mFiberCode;
        Ref<UIPropertyValue> mIOR;
        Ref<UIPropertyValue> mRBS;
        Ref<UIPropertyValue> mAttenuation;
        Ref<UIPropertyValue> mSpanOrl;
        Ref<UIPropertyValue> mSequenceType;
    };
} // namespace SE::OtdrEditor