/// @file   FileIO.h
///
/// @brief  Read and write configuration files
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2022 LeddarTech Inc. All rights reserved.

#pragma once
#include "Core/Math/Types.h"
#include "Core/String.h"
#include "yaml-cpp/yaml.h"
#include <filesystem>
#include <functional>
#include <string>

namespace fs = std::filesystem;

namespace SE::Core
{

    /// @brief Configuration file writer
    class ConfigurationWriter
    {
      public:
        /// @brief default constructor
        ConfigurationWriter() = default;

        /// @brief Write the configuration to the provided file
        ///
        /// The file is automatically written to then the object is destroyed
        ConfigurationWriter( fs::path const &aFileName );

        /// @brief Default destructor
        ~ConfigurationWriter();

        /// @brief Retrieve the written configuration as a string
        string_t GetString();

        /// @brief Create ann inlined representation of the mapping or sequence that follows
        void InlineRepresentation();

        /// @brief Begin a mapping.
        void BeginMap( bool aInline );

        /// @brief Begin a mapping. This is non-inlined.
        void BeginMap();

        /// @brief Write a key/value pair
        void WriteKey( uint32_t const &aKey )
        {
            mOut << YAML::Key << aKey << YAML::Value;
        }

        /// @brief Write a key/value pair
        void WriteKey( string_t const &aKey )
        {
            mOut << YAML::Key << aKey << YAML::Value;
        }

        /// @brief Write the value part of a key/value pair
        template <typename _Ty>
        void WriteValue( _Ty const &aValue )
        {
            mOut << aValue;
        }

        /// @brief Write the key part of a key/value pair
        template <typename _Ty>
        void WriteKey( string_t const &aKey, _Ty const &aValue )
        {
            mOut << YAML::Key << aKey << YAML::Value << aValue;
        }

        /// @brief End of mapping definition
        void EndMap();

        /// @brief Begin a sequence.
        void BeginSequence( bool aInline );

        /// @brief Begin a sequence. This is non-inlined.
        void BeginSequence();

        /// @brief End of sequence definition
        void EndSequence();

        /// @brief Write a vector as a mapping
        ///
        /// The vector is written as a mapping with the provided keys.
        ///
        /// @param aVector Vector to serialize
        /// @param aKeys Keys to use for the individual vector components
        ///
        void Write( math::vec2 const &aVector, std::array<string_t, 2> const &aKeys );

        /// @brief Write a vector as a mapping
        ///
        /// The vector is written as a mapping with the provided keys.
        ///
        /// @param aVector Vector to serialize
        /// @param aKeys Keys to use for the individual vector components
        ///
        void Write( math::vec3 const &aVector, std::array<string_t, 3> const &aKeys );

        /// @brief Write a vector as a mapping
        ///
        /// The vector is written as a mapping with the provided keys.
        ///
        /// @param aVector Vector to serialize
        /// @param aKeys Keys to use for the individual vector components
        ///
        void Write( math::vec4 const &aVector, std::array<string_t, 4> const &aKeys );
        void Write( math::quat const &aVector, std::array<string_t, 4> const &aKeys );

        template <typename _Ty>
        void Write( _Ty const &aElement )
        {
            mOut << aElement;
        }

        template <>
        void Write( math::mat4 const &aVector )
        {
            BeginSequence( true );
            for( uint32_t c = 0; c < 4; c++ )
                for( uint32_t r = 0; r < 4; r++ )
                    mOut << aVector[c][r];
            EndSequence();
        }

        /// @brief Write Null
        void WriteNull();

      private:
        fs::path      mFileName = "";
        YAML::Emitter mOut{};
    };

} // namespace SE::Core
