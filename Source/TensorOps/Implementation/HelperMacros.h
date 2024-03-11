/// @file   HelperMacros.h
///
/// @brief  Helper macros for kernel invocationa
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert. All rights reserved.

#pragma once

#include <cuda.h>
#include <curand.h>
#include <vector>

#include "../ScalarTypes.h"
#include "Core/Definitions.h"
#include "Core/Logging.h"

#define DISPATCH_BY_TYPE( type, target_fname, args ) \
    do                                               \
    {                                                \
        switch( type )                               \
        {                                            \
        case scalar_type_t::FLOAT32:                 \
            target_fname<float> args;                \
            break;                                   \
        case scalar_type_t::FLOAT64:                 \
            target_fname<double> args;               \
            break;                                   \
        case scalar_type_t::UINT8:                   \
            target_fname<uint8_t> args;              \
            break;                                   \
        case scalar_type_t::UINT16:                  \
            target_fname<uint16_t> args;             \
            break;                                   \
        case scalar_type_t::UINT32:                  \
            target_fname<uint32_t> args;             \
            break;                                   \
        case scalar_type_t::UINT64:                  \
            target_fname<uint64_t> args;             \
            break;                                   \
        case scalar_type_t::INT8:                    \
            target_fname<int8_t> args;               \
            break;                                   \
        case scalar_type_t::INT16:                   \
            target_fname<int16_t> args;              \
            break;                                   \
        case scalar_type_t::INT32:                   \
            target_fname<int32_t> args;              \
            break;                                   \
        case scalar_type_t::INT64:                   \
        default:                                     \
            target_fname<int64_t> args;              \
        }                                            \
    } while( 0 )

#define DISPATCH_BY_INTEGRAL_TYPE( aType, aTargetFname, aArgs ) \
    do                                                          \
    {                                                           \
        switch( aType )                                         \
        {                                                       \
        case scalar_type_t::UINT8:                              \
            aTargetFname<uint8_t> aArgs;                        \
            break;                                              \
        case scalar_type_t::UINT16:                             \
            aTargetFname<uint16_t> aArgs;                       \
            break;                                              \
        case scalar_type_t::UINT32:                             \
            aTargetFname<uint32_t> aArgs;                       \
            break;                                              \
        case scalar_type_t::UINT64:                             \
            aTargetFname<uint64_t> aArgs;                       \
            break;                                              \
        case scalar_type_t::INT8:                               \
            aTargetFname<int8_t> aArgs;                         \
            break;                                              \
        case scalar_type_t::INT16:                              \
            aTargetFname<int16_t> aArgs;                        \
            break;                                              \
        case scalar_type_t::INT32:                              \
            aTargetFname<int32_t> aArgs;                        \
            break;                                              \
        case scalar_type_t::INT64:                              \
        default:                                                \
            aTargetFname<int64_t> aArgs;                        \
        }                                                       \
    } while( 0 )

#define DISPATCH_BY_SIGNED_TYPE( aType, aTargetFname, aArgs ) \
    do                                                        \
    {                                                         \
        switch( aType )                                       \
        {                                                     \
        case scalar_type_t::UINT8:                            \
        case scalar_type_t::UINT16:                           \
        case scalar_type_t::UINT32:                           \
        case scalar_type_t::UINT64:                           \
            break;                                            \
        case scalar_type_t::FLOAT32:                          \
            aTargetFname<float> aArgs;                        \
            break;                                            \
        case scalar_type_t::FLOAT64:                          \
            aTargetFname<double> aArgs;                       \
            break;                                            \
        case scalar_type_t::INT8:                             \
            aTargetFname<int8_t> aArgs;                       \
            break;                                            \
        case scalar_type_t::INT16:                            \
            aTargetFname<int16_t> aArgs;                      \
            break;                                            \
        case scalar_type_t::INT32:                            \
            aTargetFname<int32_t> aArgs;                      \
            break;                                            \
        case scalar_type_t::INT64:                            \
        default:                                              \
            aTargetFname<int64_t> aArgs;                      \
        }                                                     \
    } while( 0 )

#ifndef CURAND_ASSERT
#    define CURAND_ASSERT( err ) __CURAND_ASSERT( err, __FILE__, __LINE__ )

inline void __CURAND_ASSERT( curandStatus_t err, const char *file, const int line )
{
    if( CURAND_STATUS_SUCCESS == err )
        return;

    SE::Logging::Error( "CURAND_ASSERT() API error = {} from file <{}>, line {}.\n", err, file, line );
    exit( EXIT_FAILURE );
}
#endif

namespace SE::TensorOps::Private
{
    using namespace SE::Core;

    constexpr uint32_t ThreadsPerBlock = 1024;

    template <typename _Type>
    vector_t<_Type> Resolve( vector_t<scalar_value_t> const &aValue )
    {
        vector_t<_Type> lValue( aValue.size() );
        for( uint32_t i = 0; i < aValue.size(); i++ )
        {
            lValue[i] = std::get<_Type>( aValue[i] );
        }
        return lValue;
    }
} // namespace SE::TensorOps::Private
