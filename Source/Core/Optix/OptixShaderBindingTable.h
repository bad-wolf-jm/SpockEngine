#pragma once

#include "Core/Memory.h"
#include "Optix7.h"
#include "OptixProgramGroup.h"

namespace SE::Graphics
{
    using namespace SE::Core;

    struct OptixShaderBindingTableObject
    {
        OptixShaderBindingTable RTObject;

        OptixShaderBindingTableObject() = default;

        ~OptixShaderBindingTableObject() = default;

        template <typename _RecordType>
        _RecordType NewRecordType( Ref<SE::Graphics::OptixProgramGroupObject> a_ProgramGroup )
        {
            _RecordType l_NewRecord;
            OPTIX_CHECK( optixSbtRecordPackHeader( a_ProgramGroup->RTObject, &l_NewRecord ) );
            return l_NewRecord;
        }

        template <typename _RecordType>
        std::vector<_RecordType> NewRecordType( std::vector<Ref<SE::Graphics::OptixProgramGroupObject>> a_ProgramGroup )
        {
            std::vector<_RecordType> l_NewRecordTypes;
            for( int i = 0; i < a_ProgramGroup.size(); i++ )
            {
                _RecordType rec = NewRecordType<_RecordType>( a_ProgramGroup[i] );
                l_NewRecordTypes.push_back( rec );
            }
            return l_NewRecordTypes;
        }

        void BindRayGenRecordTable( CUdeviceptr a_DevicePointer ) { RTObject.raygenRecord = a_DevicePointer; }

        template <typename _RecordType>
        void BindMissRecordTable( CUdeviceptr a_DevicePointer, size_t a_Size )
        {
            RTObject.missRecordBase          = a_DevicePointer;
            RTObject.missRecordStrideInBytes = sizeof( _RecordType );
            RTObject.missRecordCount         = static_cast<int>( a_Size );
        }

        template <typename _RecordType>
        void BindHitRecordTable( CUdeviceptr a_DevicePointer, size_t a_Size )
        {
            RTObject.hitgroupRecordBase          = a_DevicePointer;
            RTObject.hitgroupRecordStrideInBytes = sizeof( _RecordType );
            RTObject.hitgroupRecordCount         = static_cast<int>( a_Size );
        }
    };

} // namespace SE::Graphics