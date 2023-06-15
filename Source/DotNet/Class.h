#pragma once

#include <filesystem>
#include <map>
#include <string>

#include "Core/Memory.h"

#include "Instance.h"
#include "Typedefs.h"

namespace fs = std::filesystem;

namespace SE::Core
{
    class DotNetClass
    {
      public:
        DotNetClass() = default;
        DotNetClass( MonoType *aMonoClass );
        DotNetClass( const string_t &aClassNamespace, const string_t &aClassName, MonoImage *aImage, fs::path const &aDllPPath,
                     bool aIsNested = false );
        DotNetClass( MonoClass *aClass, const string_t &aClassNamespace, const string_t &aClassName, MonoImage *aImage,
                     fs::path const &aDllPPath, bool aIsNested = false );

        Ref<DotNetInstance> DoInstantiate();

        template <typename... _ArgTypes>
        Ref<DotNetInstance> Instantiate( _ArgTypes *...aArgs )
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
        MonoObject *CallMethod( const string_t &aName, _ArgTypes... aArgs )
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

        MonoMethod *GetMethod( const string_t &aName, int aParameterCount );
        MonoObject *InvokeMethod( MonoMethod *aMethod, void **aParameters );
        MonoObject *InvokeMethod( const string_t &aName, int aParameterCount, void **aParameters = nullptr );

        const std::map<string_t, sScriptField> &GetFields() const { return mFields; }

        MonoClass       *Class() { return mMonoClass; }
        sScriptProperty &GetProperty( string_t const &aName )
        {
            if( mProperties.find( aName ) != mProperties.end() )
                return mProperties[aName];
            else
                return sScriptProperty{ "", nullptr };
        }
        
        string_t &FullName() { return mClassFullName; }

        std::vector<DotNetClass *> &DerivedClasses() { return mDerived; }
        DotNetClass                *ParentClass() { return mParent; }

        operator bool() const { return ( mMonoClass != nullptr ); }

      private:
        fs::path                   mDllPath;
        DotNetClass               *mParent;
        std::vector<DotNetClass *> mDerived;

        string_t mClassNamespace;
        string_t mClassName;
        string_t mClassFullName;

        std::map<string_t, sScriptField>    mFields;
        std::map<string_t, sScriptProperty> mProperties;

        MonoClass *mMonoClass = nullptr;
        bool       mIsCore    = false;
        bool       mIsNested  = false;

        friend class DotNetRuntime;
    };

} // namespace SE::Core