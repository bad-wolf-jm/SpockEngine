/// @brief File helper functions
/// @file gli/core/file.hpp

#pragma once

#include <cstdio>

namespace numlua::core
{
    namespace detail
    {
        FILE *open_file( const char *Filename, const char *mode );
    } // namespace detail
} // namespace numlua::core

#include "./file.inl"
