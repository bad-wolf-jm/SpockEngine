/// @file   FileIO.cpp
///
/// @brief  Implementaiton file for reading and writing configuration files
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2022 LeddarTech Inc. All rights reserved.

#include "FileIO.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

using namespace std;

namespace LTSE::SensorModel
{
    namespace
    {
        static std::vector<std::string> SplitString( const std::string &aString, char aDelimiter )
        {
            vector<std::string> result;
            std::stringstream ss( aString );
            std::string item;

            while( std::getline( ss, item, aDelimiter ) )
            {
                result.push_back( item );
            }

            return result;
        }
    } // namespace
    ConfigurationWriter::ConfigurationWriter( fs::path const &aFileName )
        : mFileName{ aFileName }
    {
    }

    ConfigurationWriter::~ConfigurationWriter()
    {
        if( !mFileName.empty() )
        {
            std::ofstream lOutputFile( mFileName );
            lOutputFile << mOut.c_str();
        }
    }

    std::string ConfigurationWriter::GetString() { return std::string( mOut.c_str() ); }

    void ConfigurationWriter::InlineRepresentation() { mOut << YAML::Flow; }

    void ConfigurationWriter::BeginMap( bool aInline )
    {
        if( aInline )
            InlineRepresentation();
        mOut << YAML::BeginMap;
    }

    void ConfigurationWriter::BeginMap() { BeginMap( false ); }

    void ConfigurationWriter::EndMap() { mOut << YAML::EndMap; }

    void ConfigurationWriter::BeginSequence( bool aInline )
    {
        if( aInline )
            InlineRepresentation();
        mOut << YAML::BeginSeq;
    }
    void ConfigurationWriter::BeginSequence() { BeginSequence( false ); }

    void ConfigurationWriter::EndSequence() { mOut << YAML::EndSeq; }

    void ConfigurationWriter::Write( math::vec2 const &aVector, std::array<std::string, 2> const &aKeys )
    {
        BeginMap( true );
        WriteKey( aKeys[0], aVector.x );
        WriteKey( aKeys[1], aVector.y );
        EndMap();
    }

    void ConfigurationWriter::Write( math::vec3 const &aVector, std::array<std::string, 3> const &aKeys )
    {
        BeginMap( true );
        WriteKey( aKeys[0], aVector.x );
        WriteKey( aKeys[1], aVector.y );
        WriteKey( aKeys[2], aVector.z );
        EndMap();
    }

    void ConfigurationWriter::Write( math::vec4 const &aVector, std::array<std::string, 4> const &aKeys )
    {
        BeginMap( true );
        WriteKey( aKeys[0], aVector.x );
        WriteKey( aKeys[1], aVector.y );
        WriteKey( aKeys[2], aVector.z );
        WriteKey( aKeys[3], aVector.w );
        EndMap();
    }

    void ConfigurationWriter::WriteNull() { mOut << YAML::Null; }

    ConfigurationReader::ConfigurationReader( fs::path const &aFileName )
        : mFileName{ aFileName }
    {
        mRootNode = YAML::LoadFile( aFileName.string() );
    }

    ConfigurationReader::ConfigurationReader( std::string const &aString )
        : mFileName{ "" }
    {
        mRootNode = YAML::Load( aString );
    }

    ConfigurationReader::ConfigurationReader( YAML::Node const &aConfigurationRoot )
        : mFileName{ "" }
    {
        mRootNode = YAML::Clone( aConfigurationRoot );
    }

    ConfigurationNode::ConfigurationNode( YAML::Node const &aNode )
        : mNode{ YAML::Clone( aNode ) } {};

    ConfigurationNode ConfigurationReader::GetRoot() { return ConfigurationNode( mRootNode ); }

    bool ConfigurationNode::HasAll( std::vector<std::string> const &aKeys )
    {
        for( auto &lKey : aKeys )
        {
            if( ( ( *this )[lKey] ).IsNull() )
                return false;
        }
        return true;
    }

    ConfigurationNode ConfigurationNode::operator[]( const std::string &aKey )
    {
        if( mNode.IsNull() )
            return ConfigurationNode( YAML::Clone( mNode ) );

        std::vector<std::string> lComponents = SplitString( aKey, '.' );

        YAML::Node lCurrentNode = YAML::Clone( mNode );
        for( auto &i : lComponents )
            lCurrentNode = lCurrentNode[i];

        return ConfigurationNode( lCurrentNode );
    }

    ConfigurationNode ConfigurationNode::operator[]( const std::string &aKey ) const
    {
        if( mNode.IsNull() )
            return ConfigurationNode( YAML::Clone( mNode ) );

        std::vector<std::string> lComponents = SplitString( aKey, '.' );

        YAML::Node lCurrentNode = YAML::Clone( mNode );
        for( auto &i : lComponents )
            lCurrentNode = lCurrentNode[i];

        return ConfigurationNode( lCurrentNode );
    }

    math::vec2 ConfigurationNode::Vec( std::array<std::string, 2> const &aKeys, math::vec2 const &aDefault )
    {
        if( !HasAll( aKeys ) )
            return aDefault;

        return math::vec2{ ( ( *this )[aKeys[0]] ).As<float>( aDefault.x ), ( ( *this )[aKeys[1]] ).As<float>( aDefault.y ) };
    }

    math::vec3 ConfigurationNode::Vec( std::array<std::string, 3> const &aKeys, math::vec3 const &aDefault )
    {
        if( !HasAll( aKeys ) )
            return aDefault;

        return math::vec3{ ( *this )[aKeys[0]].As<float>( aDefault.x ), ( *this )[aKeys[1]].As<float>( aDefault.y ), ( *this )[aKeys[2]].As<float>( aDefault.z ) };
    }

    math::vec4 ConfigurationNode::Vec( std::array<std::string, 4> const &aKeys, math::vec4 const &aDefault )
    {
        if( !HasAll( aKeys ) )
            return aDefault;

        return math::vec4{ ( *this )[aKeys[0]].As<float>( aDefault.x ), ( *this )[aKeys[1]].As<float>( aDefault.y ), ( *this )[aKeys[2]].As<float>( aDefault.z ),
                           ( *this )[aKeys[3]].As<float>( aDefault.w ) };
    }

    void ConfigurationNode::ForEach( std::function<void( ConfigurationNode & )> aFunc )
    {
        for( YAML::iterator it = mNode.begin(); it != mNode.end(); ++it )
            aFunc( ConfigurationNode( *it ) );
    }

} // namespace LTSE::SensorModel
