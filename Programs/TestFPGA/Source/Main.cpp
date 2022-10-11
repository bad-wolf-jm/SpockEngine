#include "LidarSensorModel/FPGAModel/Configuration.h"

int main( int argc, char **argv )
{
    LTSE::SensorModel::sFPGAConfiguration lConfig;

    lConfig.mName    = "Cyclops";
    lConfig.mDate    = "2022-04-25";
    lConfig.mVersion = 0xB0400400;

    lConfig.mStatistics.mWindowSize = 8;

    lConfig.mStaticNoiseRemoval.mEnable               = true;
    lConfig.mStaticNoiseRemoval.mGlobalOffset         = 0;
    lConfig.mStaticNoiseRemoval.mPhotodetectorMapping = { 0,  0, 1,  0, 2,  0, 3,  0, 4,  0, 5,  0, 6,  0, 7,  0, 8,  0, 9,  0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0,
                                                          16, 0, 17, 0, 18, 0, 19, 0, 20, 0, 21, 0, 22, 0, 23, 0, 24, 0, 25, 0, 26, 0, 27, 0, 28, 0, 29, 0, 30, 0, 31, 0 };

    lConfig.mStaticNoiseRemoval.mTemplateOffsets = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    lConfig.mStaticNoiseRemoval.mTemplates       = {};

    lConfig.mBlinder.mEnable        = true;
    lConfig.mBlinder.mThreshold     = 8050;
    lConfig.mBlinder.mClipPeriod    = 1;
    lConfig.mBlinder.mWindowSize    = 3;
    lConfig.mBlinder.mBaselineDelay = 0;
    lConfig.mBlinder.mGuardLength0  = 3;
    lConfig.mBlinder.mGuardLength1  = 25;

    lConfig.mFilter.mEnable       = true;
    lConfig.mFilter.mCoefficients = { -113, 2121, 6579, 14854, 10966, 520, -2161 };

    lConfig.mCfar.mEnable          = true;
    lConfig.mCfar.mGuardLength     = 2;
    lConfig.mCfar.mWindowLength    = ( 1 << 3 );
    lConfig.mCfar.mThresholdFactor = 96;
    lConfig.mCfar.mMinStd          = 6;
    lConfig.mCfar.mSkip            = 1;

    lConfig.mPeakDetector.mEnableInterpolation = true;
    lConfig.mPeakDetector.mIgnoreCfarData      = false;
    lConfig.mPeakDetector.mMaskLength          = 2;
    lConfig.mPeakDetector.mMarginStart         = 21;
    lConfig.mPeakDetector.mMarginEnd           = 8;
    lConfig.mPeakDetector.mDetectionCount      = 5;
    lConfig.mPeakDetector.mNeighbourCount      = 5;
}