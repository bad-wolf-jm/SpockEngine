#pragma once

#include "Core/Memory.h"

#include "UI/Components/Label.h"
#include "UI/Layouts/BoxLayout.h"

namespace SE::OtdrEditor
{
    using namespace SE::Core;

    class PropertyValue : public UIBoxLayout
    {
      public:
        PropertyValue()  = default;
        ~PropertyValue() = default;

        PropertyValue( std::string aName );

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

      protected:
        Ref<UILabel> mMeasurementResultTitle;
        Ref<UIBoxLayout> mMeasurementResultLayout;
        Ref<UIBoxLayout> mMeasurementResultPropertyLayout;
        Ref<PropertyValue> mOverviewLength;
        Ref<PropertyValue> mLinkLength;
        Ref<PropertyValue> mCompletionStatus;

        Ref<UILabel> mMeasurementConfigurationTitle;
        Ref<UIBoxLayout> mMeasurementConfigurationLayout;
        Ref<UIBoxLayout> mMeasurementConfigurationPropertyLayout;

        Ref<UILabel> mAnalysisResultTitle;
        Ref<UIBoxLayout> mAnalysisResultLayout;
        Ref<UIBoxLayout> mAnalysisResultPropertyLayout;
        Ref<PropertyValue> mEOLLinkLoss;
        Ref<PropertyValue> mEOLPosition;
        Ref<PropertyValue> mProcessingTime;

        Ref<UILabel> mTestFibersTitle;

        Ref<UILabel> mSoftwareVersionsTitle;
        Ref<UIBoxLayout> mSoftwareVersionsLayout;
        Ref<UIBoxLayout> mSoftwareVersionsPropertyLayout;
        Ref<PropertyValue> mOtdrInstrumentVersion;
        Ref<PropertyValue> mOtdrSignalProcessingVersion;
        Ref<PropertyValue> mOtdrSimulationVersion;
        Ref<PropertyValue> mOlmInstrumentVersion;
        Ref<PropertyValue> mOlmSignalProcessingVersion;
        Ref<PropertyValue> mIOlmKit;

        Ref<UILabel> mHardwareTitle;
        Ref<UIBoxLayout> mHardwareLayout;
        Ref<UIBoxLayout> mHardwarePropertyLayout;
        Ref<PropertyValue> mModelName;
        Ref<PropertyValue> mSerialNumber;
        Ref<PropertyValue> mOtdrFamily;
        Ref<PropertyValue> mModelType;
        Ref<PropertyValue> mProductVersion;
        Ref<PropertyValue> mPowerLevel;
        Ref<PropertyValue> mFiberCode;
        Ref<PropertyValue> mOutputPorts;
        Ref<PropertyValue> mFPGAversion;
        Ref<PropertyValue> mPCBVersion;
        Ref<PropertyValue> mPCBSerialNumber;
        Ref<PropertyValue> mCPUPCBVersion;
        Ref<PropertyValue> mCPUPCBSerialNumber;
        Ref<PropertyValue> mPowerMeterVersion;
        Ref<PropertyValue> mPowerMeterSerialNumber;
        Ref<PropertyValue> mManufacturingDate;
        Ref<PropertyValue> mLastCalibrationDate;
        Ref<PropertyValue> mUserLastCalibrationDate;
    };
} // namespace SE::OtdrEditor