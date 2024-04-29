/// @brief Include to compute the number of mipmaps levels necessary to create a mipmap complete texture.
/// @file gli/levels.hpp

#pragma once

#include "type.hpp"

namespace numlua::core
{
    /// Compute the number of mipmaps levels necessary to create a mipmap complete texture
    ///
    /// @param Extent Extent of the texture base level mipmap
    /// @tparam vecType Vector type used to express the dimensions of a texture of any kind.
    /// @code
    /// #include <gli/texture2d.hpp>
    /// #include <gli/levels.hpp>
    /// ...
    /// numlua::core::texture2d::extent_type Extent(32, 10);
    /// numlua::core::texture2d Texture(numlua::core::levels(Extent));
    /// @endcode
    template <length_t L, typename T, qualifier P>
    T levels( vec<L, T, P> const &Extent );
    /*
        /// Compute the number of mipmaps levels necessary to create a mipmap complete texture
        ///
        /// @param Extent Extent of the texture base level mipmap
        /// @code
        /// #include <gli/texture2d.hpp>
        /// #include <gli/levels.hpp>
        /// ...
        /// numlua::core::texture2d Texture(32);
        /// @endcode
        size_t levels(size_t Extent);

        /// Compute the number of mipmaps levels necessary to create a mipmap complete texture
        ///
        /// @param Extent Extent of the texture base level mipmap
        /// @code
        /// #include <gli/texture2d.hpp>
        /// #include <gli/levels.hpp>
        /// ...
        /// numlua::core::texture2d Texture(32);
        /// @endcode
        int levels(int Extent);
    */
} // namespace numlua::core

#include "./core/levels.inl"
