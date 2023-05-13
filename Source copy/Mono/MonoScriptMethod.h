#pragma once

#include <filesystem>
#include <map>
#include <string>


#include "MonoScriptInstance.h"
#include "MonoTypedefs.h"

namespace SE::Core
{
    class MonoScriptInstance;

    class MonoScriptMehod
    {
      public:
        MonoScriptMehod() = default;
        MonoScriptMehod( MonoMethod *aMonoMethod, MonoScriptInstance *aInstance );

        template <typename... _ArgTypes>
        MonoObject *operator()( _ArgTypes... aArgs )
        {
            void *lParameters[] = { (void *)&aArgs... };

            return mInstance->InvokeMethod( mMonoMethod, lParameters );
        }

      private:
        MonoMethod         *mMonoMethod;
        MonoScriptInstance *mInstance;
    };
} // namespace SE::Core