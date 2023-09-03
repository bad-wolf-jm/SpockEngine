#pragma once

#include "Core/String.h"

using namespace SE::Core;

extern "C"
{
    typedef struct _MonoClass      MonoClass;
    typedef struct _MonoMethod     MonoMethod;
    typedef struct _MonoString     MonoString;
    typedef struct _MonoArray      MonoArray;
    typedef struct _MonoType       MonoType;
    typedef struct _MonoObject     MonoObject;
    typedef struct _MonoMethod     MonoMethod;
    typedef struct _MonoAssembly   MonoAssembly;
    typedef struct _MonoImage      MonoImage;
    typedef struct _MonoClassField MonoClassField;
    typedef struct _MonoProperty   MonoProperty;

    enum class eScriptFieldType
    {
        None = 0,
        Float,
        Double,
        Bool,
        Char,
        Byte,
        Short,
        Int,
        Long,
        UByte,
        UShort,
        UInt,
        ULong
    };

    struct sScriptField
    {
        eScriptFieldType mType;
        string_t      mName;
        MonoClassField  *mClassField;
    };

    struct sScriptProperty
    {
        string_t   mName;
        MonoProperty *mProperty;

        operator bool() { return mProperty != nullptr; }
    };
}