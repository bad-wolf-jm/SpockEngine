#pragma once

#include "Class.h"
#include "Typedefs.h"
#include <filesystem>

namespace SE::Core::Mono::Utils
{
    char            *ReadBytes( const std::filesystem::path &aFilepath, uint32_t *aOutSize );
    MonoAssembly    *LoadMonoAssembly( const std::filesystem::path &lAssemblyPath );
    void             PrintAssemblyTypes( MonoAssembly *aAssembly );
    eScriptFieldType MonoTypeToScriptFieldType( MonoType *aMonoType );
    string_t      GetClassFullName( const char *aNameSpace, const char *aClassName );
    string_t      GetClassFullName( MonoClass *aClass );

} // namespace SE::Core::Mono::Utils
