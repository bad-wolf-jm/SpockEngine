/// @file   Core.h
///
/// @brief  Global definitions
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#pragma once

#ifdef _MSC_VER
#    define LOG2( X ) ( (unsigned)( 8 * sizeof( uint64_t ) - __lzcnt64( ( (uint64_t)X ) ) - 1 ) )
#else
#    define LOG2( X ) ( (unsigned)( 8 * sizeof( uint64_t ) - __builtin_clzll( ( (uint64_t)X ) ) - 1 ) )
#endif
