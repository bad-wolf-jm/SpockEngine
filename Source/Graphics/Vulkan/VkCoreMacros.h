#pragma once

#include "Core/Core.h"
#include "Core/Logging.h"
#include <fmt/core.h>
#include <stdexcept>

#ifndef VK_CHECK_RESULT
#    define VK_CHECK_RESULT( err ) __VK_ASSERT( (VkResult)err, __FILE__, __LINE__ )

inline void __VK_ASSERT( VkResult aErr, const char *aFile, const int aLine )
{
    if( aErr != VK_SUCCESS )
    {
        std::string lErrorString = fmt::format( "Fatal : VkResult is \"{}\" in {} at line {}", aErr, aFile, aLine );
        SE::Logging::Error( lErrorString );

        throw std::runtime_error( lErrorString );
    }
}
#endif

#define VK_SAMPLE_COUNT_VALUE( x ) ( static_cast<VkSampleCountFlagBits>( 1 << LOG2( x ) ) )
