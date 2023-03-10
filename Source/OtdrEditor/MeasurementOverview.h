#pragma once

#include "Core/Memory.h"

#include "UI/Components/Label.h"
#include "UI/Layouts/BoxLayout.h"

#include "Mono/MonoScriptInstance.h"

namespace SE::OtdrEditor
{
    using namespace SE::Core;

    class UIPropertyValue : public UIBoxLayout
    {
      public:
        UIPropertyValue()  = default;
        ~UIPropertyValue() = default;

        UIPropertyValue( std::string aName );

        void SetValue( std::string aValue );

      protected:
        Ref<UILabel> mName;
        Ref<UILabel> mValue;
    };

    class MeasurementOverview : public UIBoxLayout
    {
      public:
        MeasurementOverview();
        ~MeasurementOverview() = default;

        void SetData(Ref<MonoScriptInstance> aMeasurementOverview);

      protected:
        Ref<UILabel>         mMeasurementResultTitle;
        Ref<UIBoxLayout>     mMeasurementResultLayout;
        Ref<UIBoxLayout>     mMeasurementResultPropertyLayout;
        Ref<UIPropertyValue> mOverviewLength;
        Ref<UIPropertyValue> mLinkLength;
        Ref<UIPropertyValue> mCompletionStatus;

        Ref<UILabel>         mMeasurementConfigurationTitle;
        Ref<UIBoxLayout>     mMeasurementConfigurationLayout;
        Ref<UIBoxLayout>     mMeasurementConfigurationPropertyLayout;
        Ref<UIPropertyValue> mLaunchFiberLength;
        Ref<UIPropertyValue> mConditioner;
        Ref<UIPropertyValue> mReceiveFiberLength;
        Ref<UIPropertyValue> mLoopLength;
        Ref<UIPropertyValue> mConfigurationFiberCode;
        Ref<UIPropertyValue> mIOR;
        Ref<UIPropertyValue> mRBS;
        Ref<UIPropertyValue> mAttenuation;
        Ref<UIPropertyValue> mWavelengths;
        Ref<UIPropertyValue> mDirection;
        Ref<UIPropertyValue> mLoopbackExtract;
        Ref<UIPropertyValue> mExpectedLinkElements;
        Ref<UIPropertyValue> mTopology;
        Ref<UIPropertyValue> mMeasurementType;
        Ref<UIPropertyValue> mSequenceType;
        Ref<UIPropertyValue> mMacrobendThreshold;
        Ref<UIPropertyValue> mComment;
        Ref<UIPropertyValue> mAdvisorMode;
        Ref<UIPropertyValue> mDuration;
        Ref<UIPropertyValue> mDate;

        Ref<UILabel>         mAnalysisResultTitle;
        Ref<UIBoxLayout>     mAnalysisResultLayout;
        Ref<UIBoxLayout>     mAnalysisResultPropertyLayout;
        Ref<UIPropertyValue> mEOLLinkLoss;
        Ref<UIPropertyValue> mEOLPosition;
        Ref<UIPropertyValue> mProcessingTime;

        Ref<UILabel> mTestFibersTitle;

        Ref<UILabel>         mSoftwareVersionsTitle;
        Ref<UIBoxLayout>     mSoftwareVersionsLayout;
        Ref<UIBoxLayout>     mSoftwareVersionsPropertyLayout;
        Ref<UIPropertyValue> mOtdrInstrumentVersion;
        Ref<UIPropertyValue> mOtdrSignalProcessingVersion;
        Ref<UIPropertyValue> mOtdrSimulationVersion;
        Ref<UIPropertyValue> mOlmInstrumentVersion;
        Ref<UIPropertyValue> mOlmSignalProcessingVersion;
        Ref<UIPropertyValue> mIOlmKit;

        Ref<UILabel>         mHardwareTitle;
        Ref<UIBoxLayout>     mHardwareLayout;
        Ref<UIBoxLayout>     mHardwarePropertyLayout;
        Ref<UIPropertyValue> mModelName;
        Ref<UIPropertyValue> mSerialNumber;
        Ref<UIPropertyValue> mOtdrFamily;
        Ref<UIPropertyValue> mModelType;
        Ref<UIPropertyValue> mProductVersion;
        Ref<UIPropertyValue> mPowerLevel;
        Ref<UIPropertyValue> mFiberCode;
        Ref<UIPropertyValue> mOutputPorts;
        Ref<UIPropertyValue> mFPGAversion;
        Ref<UIPropertyValue> mPCBVersion;
        Ref<UIPropertyValue> mPCBSerialNumber;
        Ref<UIPropertyValue> mCPUPCBVersion;
        Ref<UIPropertyValue> mCPUPCBSerialNumber;
        Ref<UIPropertyValue> mPowerMeterVersion;
        Ref<UIPropertyValue> mPowerMeterSerialNumber;
        Ref<UIPropertyValue> mManufacturingDate;
        Ref<UIPropertyValue> mLastCalibrationDate;
        Ref<UIPropertyValue> mUserLastCalibrationDate;
    };
} // namespace SE::OtdrEditor