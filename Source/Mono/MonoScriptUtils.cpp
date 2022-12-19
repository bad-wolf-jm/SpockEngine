#pragma once
#include "MonoScriptUtils.h"
#include <filesystem>
#include <unordered_map>
#include <string>
#include <fstream>

#include "Core/Logging.h"

#include "mono/jit/jit.h"
#include "mono/metadata/assembly.h"
#include "mono/metadata/object.h"
#include "mono/metadata/tabledefs.h"

namespace SE::Core::Mono::Utils
{
    static std::unordered_map<std::string, eScriptFieldType> sScriptFieldTypeMap = {
        { "System.Single", eScriptFieldType::Float },  { "System.Double", eScriptFieldType::Double },
        { "System.Boolean", eScriptFieldType::Bool },  { "System.Char", eScriptFieldType::Char },
        { "System.Int16", eScriptFieldType::Short },   { "System.Int32", eScriptFieldType::Int },
        { "System.Int64", eScriptFieldType::Long },    { "System.Byte", eScriptFieldType::Byte },
        { "System.UInt16", eScriptFieldType::UShort }, { "System.UInt32", eScriptFieldType::UInt },
        { "System.UInt64", eScriptFieldType::ULong } };

    char *ReadBytes( const std::filesystem::path &aFilepath, uint32_t *aOutSize )
    {
        std::ifstream lStream( aFilepath, std::ios::binary | std::ios::ate );

        if( !lStream ) return nullptr;

        std::streampos end = lStream.tellg();
        lStream.seekg( 0, std::ios::beg );
        uint64_t size = end - lStream.tellg();

        if( size == 0 ) return nullptr;

        char *buffer = new char[size];
        lStream.read( (char *)buffer, size );
        lStream.close();

        *aOutSize = (uint32_t)size;
        return buffer;
    }

    MonoAssembly *LoadMonoAssembly( const std::filesystem::path &lAssemblyPath )
    {
        uint32_t lFileSize = 0;
        char    *lFileData = ReadBytes( lAssemblyPath, &lFileSize );

        MonoImageOpenStatus lStatus;
        MonoImage          *lImage = mono_image_open_from_data_full( lFileData, lFileSize, 1, &lStatus, 0 );

        if( lStatus != MONO_IMAGE_OK )
        {
            const char *lErrorMessage = mono_image_strerror( lStatus );
            return nullptr;
        }

        std::string   lPathString = lAssemblyPath.string();
        MonoAssembly *lAssembly   = mono_assembly_load_from_full( lImage, lPathString.c_str(), &lStatus, 0 );
        mono_image_close( lImage );

        delete[] lFileData;

        return lAssembly;
    }

    void PrintAssemblyTypes( MonoAssembly *aAssembly )
    {
        MonoImage           *lImage                = mono_assembly_get_image( aAssembly );
        const MonoTableInfo *lTypeDefinitionsTable = mono_image_get_table_info( lImage, MONO_TABLE_TYPEDEF );
        int32_t              lTypesCount           = mono_table_info_get_rows( lTypeDefinitionsTable );

        for( int32_t i = 0; i < lTypesCount; i++ )
        {
            uint32_t lCols[MONO_TYPEDEF_SIZE];
            mono_metadata_decode_row( lTypeDefinitionsTable, i, lCols, MONO_TYPEDEF_SIZE );

            const char *lNameSpace = mono_metadata_string_heap( lImage, lCols[MONO_TYPEDEF_NAMESPACE] );
            const char *lName      = mono_metadata_string_heap( lImage, lCols[MONO_TYPEDEF_NAME] );
            SE::Logging::Info( "{}.{}", lNameSpace, lName );
        }
    }

    eScriptFieldType MonoTypeToScriptFieldType( MonoType *aMonoType )
    {
        std::string typeName = mono_type_get_name( aMonoType );

        auto it = sScriptFieldTypeMap.find( typeName );
        if( it == sScriptFieldTypeMap.end() ) return eScriptFieldType::None;

        return it->second;
    }
} // namespace Mono::Utils
