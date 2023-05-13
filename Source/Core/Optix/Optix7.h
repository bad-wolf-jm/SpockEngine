// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

// optix 7
#include "Core/CUDA/Cuda.h"
#include <optix.h>
#include <optix_stubs.h>
#include <sstream>
#include <stdexcept>

#define CUDA_CHECK( call )                                                                               \
    {                                                                                                    \
        cudaError_t rc = cuda##call;                                                                     \
        if( rc != cudaSuccess )                                                                          \
        {                                                                                                \
            std::stringstream txt;                                                                       \
            cudaError_t       err = rc; /*cudaGetLastError();*/                                          \
            txt << "CUDA Error " << cudaGetErrorName( err ) << " (" << cudaGetErrorString( err ) << ")"; \
            throw std::runtime_error( txt.str() );                                                       \
        }                                                                                                \
    }

#define CUDA_CHECK_NOEXCEPT( call ) \
    {                               \
        cuda##call;                 \
    }

#define OPTIX_CHECK( call )                                                                             \
    {                                                                                                   \
        OptixResult res = call;                                                                         \
        if( res != OPTIX_SUCCESS )                                                                      \
        {                                                                                               \
            fprintf( stderr, "Optix call (%s) failed with code %d (line %d)\n", #call, res, __LINE__ ); \
            throw std::runtime_error( "" );                                                             \
        }                                                                                               \
    }

#define OPTIX_CHECK_NO_EXCEPT( call )                                                                   \
    {                                                                                                   \
        OptixResult res = call;                                                                         \
        if( res != OPTIX_SUCCESS )                                                                      \
        {                                                                                               \
            fprintf( stderr, "Optix call (%s) failed with code %d (line %d)\n", #call, res, __LINE__ ); \
        }                                                                                               \
    }