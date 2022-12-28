#pragma once

#include <filesystem>
#include <map>
#include <string>

#include "MonoTypedefs.h"
#include "MonoScriptInstance.h"

namespace SE::Core
{
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
        std::string      mName;
        MonoClassField  *mClassField;
    };

    class MonoScriptClass
    {
      public:
        MonoScriptClass() = default;
        MonoScriptClass( MonoType *aMonoClass );
        MonoScriptClass( const std::string &aClassNamespace, const std::string &aClassName, bool aIsCore = false );

        MonoScriptInstance Instantiate();

        template <typename... _ArgTypes>
        MonoScriptInstance Instantiate( _ArgTypes... aArgs )
        {
            void *lParameters[] = { (void *)&aArgs... };

            auto lNewInstance = Instantiate();
            lNewInstance.InvokeMethod( ".ctor", sizeof...( _ArgTypes ), lParameters );

            return lNewInstance;
        }

        template <typename... _ArgTypes>
        MonoObject *CallMethod( const std::string &aName, _ArgTypes... aArgs )
        {
            void *lParameters[] = { (void *)&aArgs... };

            return InvokeMethod( aName, sizeof...( _ArgTypes ), lParameters );
        }

        MonoMethod *GetMethod( const std::string &aName, int aParameterCount );
        MonoObject *InvokeMethod( MonoMethod *aMethod, void **aParameters );
        MonoObject *InvokeMethod( const std::string &aName, int aParameterCount, void **aParameters = nullptr );

        const std::map<std::string, sScriptField> &GetFields() const { return mFields; }

        MonoClass * Class() { return mMonoClass; }

      private:
        std::string mClassNamespace;
        std::string mClassName;

        std::map<std::string, sScriptField> mFields;

        MonoClass *mMonoClass = nullptr;
        bool       mIsCore    = false;

        friend class MonoScriptEngine;
    };

} // namespace SE::Core