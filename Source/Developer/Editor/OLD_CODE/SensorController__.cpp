#include "SensorController.h"

namespace LTSE::Editor
{
    SensorController::SensorController( Ref<SensorDeviceBase> a_ControlledSensor )
        : m_ControlledSensor{ a_ControlledSensor }
    {
    }

    Ref<EnvironmentSampler> SensorController::Sample()
    {
        if( CurrentTileLayout )
        {
            auto& lTileLayout = CurrentTileLayout.Get<sTileLayoutComponent>().mLayout;
            std::vector<std::string> lTileIds = {};
            std::vector<math::vec2> lTilePositions = {};
            std::vector<float> lTileTimes = {};
            for (auto & x : lTileLayout)
            {
                lTileIds.push_back(x.second.first);
                lTilePositions.push_back(x.second.second);
                lTileTimes.push_back(0.0f);
            }

            AcquisitionSpecification lAcqCreateInfo{};
            lAcqCreateInfo.mBasePoints   = 100;
            lAcqCreateInfo.mOversampling = 1;

            return m_ControlledSensor->Sample( EnvironmentSamplingParameter, lAcqCreateInfo, lTileIds, lTilePositions, lTileTimes );
        }
        else if (RunSensorSimulation)
        {
            return nullptr;
        }
        return nullptr;
    }

} // namespace LTSE::Editor