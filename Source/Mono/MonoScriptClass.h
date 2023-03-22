#pragma once

#include <filesystem>
#include <map>
#include <string>

#include "Core/Memory.h"

#include "MonoScriptInstance.h"
#include "MonoTypedefs.h"

namespace fs = std::filesystem;

namespace SE::Core
{
    class MonoScriptClass
    {
      public:
        MonoScriptClass() = default;
        MonoScriptClass( MonoType *aMonoClass );
        MonoScriptClass( const std::string &aClassNamespace, const std::string &aClassName, MonoImage *aImage,
                         fs::path const &aDllPPath, bool aIsNested = false );
        MonoScriptClass( MonoClass *aClass, const std::string &aClassNamespace, const std::string &aClassName, MonoImage *aImage,
                         fs::path const &aDllPPath, bool aIsNested = false );

        Ref<MonoScriptInstance> DoInstantiate();

        template <typename... _ArgTypes>
        Ref<MonoScriptInstance> Instantiate( _ArgTypes *...aArgs )
        {
            auto lNewInstance = DoInstantiate();

            if constexpr( sizeof...( _ArgTypes ) != 0 )
            {
                void *lParameters[] = { (void *)aArgs... };

                lNewInstance->InvokeMethod( ".ctor", sizeof...( _ArgTypes ), lParameters );
            }
            else
            {
                lNewInstance->InvokeMethod( ".ctor", 0, NULL );
            }

            return lNewInstance;
        }

        template <typename... _ArgTypes>
        MonoObject *CallMethod( const std::string &aName, _ArgTypes... aArgs )
        {
            if constexpr( sizeof...( _ArgTypes ) == 0 )
            {
                return InvokeMethod( aName, 0, nullptr );
            }
            else
            {
                void *lParameters[] = { (void *)aArgs... };

                return InvokeMethod( aName, sizeof...( _ArgTypes ), lParameters );
            }
        }

        MonoMethod *GetMethod( const std::string &aName, int aParameterCount );
        MonoObject *InvokeMethod( MonoMethod *aMethod, void **aParameters );
        MonoObject *InvokeMethod( const std::string &aName, int aParameterCount, void **aParameters = nullptr );

        const std::map<std::string, sScriptField> &GetFields() const { return mFields; }

        MonoClass       *Class() { return mMonoClass; }
        sScriptProperty &GetProperty( std::string const &aName )
        {
            if( mProperties.find( aName ) != mProperties.end() )
                return mProperties[aName];
            else
                return sScriptProperty{ "", nullptr };
        }
        std::string &FullName() { return mClassFullName; }

        std::vector<MonoScriptClass *> &DerivedClasses() { return mDerived; }
        MonoScriptClass                *ParentClass() { return mParent; }

      private:
        fs::path                       mDllPath;
        MonoScriptClass               *mParent;
        std::vector<MonoScriptClass *> mDerived;

        std::string mClassNamespace;
        std::string mClassName;
        std::string mClassFullName;

        std::map<std::string, sScriptField>    mFields;
        std::map<std::string, sScriptProperty> mProperties;

        MonoClass *mMonoClass = nullptr;
        bool       mIsCore    = false;
        bool       mIsNested  = false;

        friend class MonoRuntime;
    };

} // namespace SE::Core