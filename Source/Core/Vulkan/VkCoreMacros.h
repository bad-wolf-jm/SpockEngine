#pragma once

#include "Core/Core.h"
#include "Core/Logging.h"
#include <fmt/core.h>
#include <stdexcept>

#define VK_CHECK_RESULT( f )                                                                                 \
    do                                                                                                       \
    {                                                                                                        \
        VkResult lVkResult = ( f );                                                                          \
        if( lVkResult != VK_SUCCESS )                                                                        \
        {                                                                                                    \
            std::string l_ErrorString =                                                                      \
                fmt::format( "Fatal : VkResult is \"{}\" in {} at line {}", lVkResult, __FILE__, __LINE__ ); \
            LTSE::Logging::Error( l_ErrorString );                                                           \
            throw std::runtime_error( l_ErrorString );                                                       \
        }                                                                                                    \
    } while( 0 )

#define VK_SAMPLE_COUNT_VALUE( x ) ( static_cast<VkSampleCountFlagBits>( 1 << LOG2( x ) ) )
