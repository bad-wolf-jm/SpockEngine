#include "Configuration.h"

#include <fstream>
#include <iterator>

namespace LTSE::SensorModel
{
    std::vector<std::vector<float>> sFPGAConfiguration::GetStaticNoiseTemplates( sTileConfiguration const &aTileConfiguration, uint32_t aTraceLength ) const
    {
        constexpr uint32_t lHeaderSize     = 16;
        constexpr uint32_t lDatasetLength  = 32;
        constexpr uint32_t lTemplateLength = 64;

        std::vector<std::vector<float>> lTemplates;
        for( auto &i : aTileConfiguration.mLCA3PhotodetectorIndices )
        {
            uint32_t lTemplateStartOffset = mStaticNoiseRemoval.mTemplateOffsets[i] + mStaticNoiseRemoval.mGlobalOffset;

            uint32_t lTemplateStreamOffset = lHeaderSize + ( mStaticNoiseRemoval.mDatasetSelector * lDatasetLength * lTemplateLength ) +
                                             ( mStaticNoiseRemoval.mPhotoDetectorMapping[i] * lTemplateLength ) + lTemplateStartOffset;

            std::vector<float> lNewTemplate( aTraceLength, 0 );

            for( uint32_t lSampleIdx = 0; lSampleIdx < lTemplateLength; lSampleIdx++ )
                lNewTemplate[lSampleIdx] = static_cast<float>( aTileConfiguration.mStaticNoiseTemplateData[lTemplateStreamOffset + lSampleIdx] );

            lTemplates.push_back( lNewTemplate );
        }

        return lTemplates;
    }

    uint32_t sFPGAConfiguration::GetHeaderSize( sTileConfiguration const &aTileConfiguration ) const { return 5; }

    uint32_t sFPGAConfiguration::GetPacketLength( sTileConfiguration const &aTileConfiguration ) const
    {
        const uint32_t lWaveformPacketHeaderLength = 4;

        uint32_t lShortWaveformLength = 0;

        if( aTileConfiguration.mPeakDetectorMode == sTileConfiguration::PeakDetectorMode::COMPRESSED_RAW_DATA )
            lShortWaveformLength = 2 * aTileConfiguration.mNeighbourCount + 1;
        return lWaveformPacketHeaderLength + lShortWaveformLength;
    }

} // namespace LTSE::SensorModel
