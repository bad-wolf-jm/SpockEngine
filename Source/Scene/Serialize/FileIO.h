/// @file   FileIO.h
///
/// @brief  Read and write configuration files
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2022 LeddarTech Inc. All rights reserved.

#pragma once
#include "Core/Math/Types.h"
#include "yaml-cpp/yaml.h"
#include <filesystem>
#include <functional>
#include <string>

namespace fs = std::filesystem;

namespace SE::Core
{
    /// @brief Abstraction around an entry in the configuration file.
    class ConfigurationNode
    {
      public:
        /// @brief Constructs a configuration from a YAML node
        ConfigurationNode() = default;

        /// @brief Constructs a configuration from a YAML node
        ConfigurationNode( YAML::Node const &aNode );

        /// @brief Default destructor
        ~ConfigurationNode() = default;

      public:
        /// @brief Parse the node's contents into a vector
        ///
        /// The node is assumed to point to a dictionary. The provided keys will be used to fill the vector.
        /// If any of the keys is missing, then the default value is returned.
        ///
        /// @param aKeys Keys to use to build the vector
        /// @param aDefault Default vector to return if a key is missing
        ///
        math::vec2 Vec( std::array<std::string, 2> const &aKeys, math::vec2 const &aDefault );

        /// @brief Parse the node's contents into a vector
        ///
        /// The node is assumed to represent a dictionary. The provided keys will be used to fill the vector.
        /// If any of the keys is missing, then the default value is returned.
        ///
        /// @param aKeys Keys to use to build the vector
        /// @param aDefault Default vector to return if a key is missing
        ///
        math::vec3 Vec( std::array<std::string, 3> const &aKeys, math::vec3 const &aDefault );

        /// @brief Parse the node's contents into a vector
        ///
        /// The node is assumed to represent a dictionary. The provided keys will be used to fill the vector.
        /// If any of the keys is missing, then the default value is returned.
        ///
        /// @param aKeys Keys to use to build the vector
        /// @param aDefault Default vector to return if a key is missing
        ///
        math::vec4 Vec( std::array<std::string, 4> const &aKeys, math::vec4 const &aDefault );

        /// @brief Test whether the underlying node is Null
        bool IsNull() { return mNode.IsNull(); }

        /// @brief Test whether the underlying node has a given set of keys
        bool HasAll( std::vector<std::string> const &aKeys );

        template <uint32_t N>
        bool HasAll( std::array<std::string, N> const &aKeys )
        {
            for( auto &lKey : aKeys )
            {
                if( ( ( *this )[lKey] ).IsNull() ) return false;
            }
            return true;
        }

        /// @brief Interpret the value of the node as the provided type
        template <typename _Ty>
        _Ty As( _Ty aDefault )
        {
            if( mNode.IsNull() ) return aDefault;

            return mNode.as<_Ty>();
        }

        /// @brief Retrieve a child by name
        ///
        /// This assumed that the current node is a mapping. The key is a dot-separated path in the node hierarchy.
        /// If there is a non-existing ID in the path, then we return a null node.
        ///
        /// @param aKey Key to retrieve
        ///
        /// @return The node
        ///
        ConfigurationNode operator[]( std::string const &aKey );

        /// @brief Retrieve a child by name (const version)
        ///
        /// This assumed that the current node is a mapping. The key is a dot-separated path in the node hierarchy.
        /// If there is a non-existing ID in the path, then we return a null node.
        ///
        /// @param aKey Key to retrieve
        ///
        /// @return The node
        ///
        ConfigurationNode operator[]( std::string const &aKey ) const;

        ConfigurationNode &operator=( ConfigurationNode const &aKey );

        // /// @brief Iterate over a mapping
        // template <typename _KeyType>
        // void ForEach( std::function<void( _KeyType const &, ConfigurationNode & )> aFunc ) const
        // {
        //     for( YAML::const_iterator it = mNode.begin(); it != mNode.end(); ++it )
        //     {
        //         YAML::Node const &lCurrentTexture = it->second;
        //         aFunc( it->first.as<_KeyType>(), ConfigurationNode( it->second ) );
        //     }
        // }

        /// @brief Iterate over a mapping
        template <typename _KeyType>
        void ForEach( std::function<void( _KeyType const &, ConfigurationNode const & )> aFunc ) const
        {
            for( YAML::const_iterator it = mNode.begin(); it != mNode.end(); ++it )
            {
                YAML::Node const &lCurrentTexture = it->second;
                aFunc( it->first.as<_KeyType>(), ConfigurationNode( it->second ) );
            }
        }

        /// @brief Iterate over a list
        void ForEach( std::function<void( ConfigurationNode & )> aFunc );

      private:
        YAML::Node mNode{}; //!< Underlying node
    };

    /// @brief Abstract configuration reader
    class ConfigurationReader
    {
      public:
        /// @brief Read the configuration from a file
        ///
        /// @param aFileName Path containing the YAML description of the sensor to load
        ///
        ConfigurationReader( fs::path const &aFileName );

        /// @brief Read the configuration from a string
        ///
        /// @param aString String containing the YAML description of the sensor to load
        ///
        ConfigurationReader( std::string const &aString );

        /// @brief Read the configuration from a pre-parsed YAML node
        ConfigurationReader( YAML::Node const &aConfigurationRoot );

        /// @brief Default destructor
        ~ConfigurationReader() = default;

        /// @brief Retrieve the root node of the file as our internal node type. This is the entry point
        ConfigurationNode GetRoot();

      private:
        fs::path   mFileName = "";
        YAML::Node mRootNode{};
    };

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
        std::string GetString();

        /// @brief Create ann inlined representation of the mapping or sequence that follows
        void InlineRepresentation();

        /// @brief Begin a mapping.
        void BeginMap( bool aInline );

        /// @brief Begin a mapping. This is non-inlined.
        void BeginMap();

        /// @brief Write a key/value pair
        void WriteKey( uint32_t const &aKey ) { mOut << YAML::Key << aKey << YAML::Value; }

        /// @brief Write a key/value pair
        void WriteKey( std::string const &aKey ) { mOut << YAML::Key << aKey << YAML::Value; }

        /// @brief Write the value part of a key/value pair
        template <typename _Ty>
        void WriteValue( _Ty const &aValue )
        {
            mOut << aValue;
        }

        /// @brief Write the key part of a key/value pair
        template <typename _Ty>
        void WriteKey( std::string const &aKey, _Ty const &aValue )
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
        void Write( math::vec2 const &aVector, std::array<std::string, 2> const &aKeys );

        /// @brief Write a vector as a mapping
        ///
        /// The vector is written as a mapping with the provided keys.
        ///
        /// @param aVector Vector to serialize
        /// @param aKeys Keys to use for the individual vector components
        ///
        void Write( math::vec3 const &aVector, std::array<std::string, 3> const &aKeys );

        /// @brief Write a vector as a mapping
        ///
        /// The vector is written as a mapping with the provided keys.
        ///
        /// @param aVector Vector to serialize
        /// @param aKeys Keys to use for the individual vector components
        ///
        void Write( math::vec4 const &aVector, std::array<std::string, 4> const &aKeys );
        void Write( math::quat const &aVector, std::array<std::string, 4> const &aKeys );

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
                for( uint32_t r = 0; r < 4; r++ ) mOut << aVector[c][r];
            EndSequence();
        }

        /// @brief Write Null
        void WriteNull();

      private:
        fs::path      mFileName = "";
        YAML::Emitter mOut{};
    };

} // namespace SE::Core
