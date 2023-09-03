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

namespace SE::Core
{
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

    string_t ConfigurationWriter::GetString()
    {
        return string_t( mOut.c_str() );
    }

    void ConfigurationWriter::InlineRepresentation()
    {
        mOut << YAML::Flow;
    }

    void ConfigurationWriter::BeginMap( bool aInline )
    {
        if( aInline )
            InlineRepresentation();
        mOut << YAML::BeginMap;
    }

    void ConfigurationWriter::BeginMap()
    {
        BeginMap( false );
    }

    void ConfigurationWriter::EndMap()
    {
        mOut << YAML::EndMap;
    }

    void ConfigurationWriter::BeginSequence( bool aInline )
    {
        if( aInline )
            InlineRepresentation();
        mOut << YAML::BeginSeq;
    }
    void ConfigurationWriter::BeginSequence()
    {
        BeginSequence( false );
    }

    void ConfigurationWriter::EndSequence()
    {
        mOut << YAML::EndSeq;
    }

    void ConfigurationWriter::Write( math::vec2 const &aVector, std::array<string_t, 2> const &aKeys )
    {
        BeginMap( true );
        WriteKey( aKeys[0], aVector.x );
        WriteKey( aKeys[1], aVector.y );
        EndMap();
    }

    void ConfigurationWriter::Write( math::vec3 const &aVector, std::array<string_t, 3> const &aKeys )
    {
        BeginMap( true );
        WriteKey( aKeys[0], aVector.x );
        WriteKey( aKeys[1], aVector.y );
        WriteKey( aKeys[2], aVector.z );
        EndMap();
    }

    void ConfigurationWriter::Write( math::vec4 const &aVector, std::array<string_t, 4> const &aKeys )
    {
        BeginMap( true );
        WriteKey( aKeys[0], aVector.x );
        WriteKey( aKeys[1], aVector.y );
        WriteKey( aKeys[2], aVector.z );
        WriteKey( aKeys[3], aVector.w );
        EndMap();
    }

    void ConfigurationWriter::Write( math::quat const &aVector, std::array<string_t, 4> const &aKeys )
    {
        BeginMap( true );
        WriteKey( aKeys[0], aVector.x );
        WriteKey( aKeys[1], aVector.y );
        WriteKey( aKeys[2], aVector.z );
        WriteKey( aKeys[3], aVector.w );
        EndMap();
    }

    void ConfigurationWriter::WriteNull()
    {
        mOut << YAML::Null;
    }

} // namespace SE::Core
