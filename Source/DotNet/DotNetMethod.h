#pragma once

#include <filesystem>
#include <map>
#include <string>


#include "DotNetInstance.h"
#include "Typedefs.h"

namespace SE::Core
{
    class DotNetInstance;

    class DotNetMehod
    {
      public:
        DotNetMehod() = default;
        DotNetMehod( MonoMethod *aMonoMethod, DotNetInstance *aInstance );

        template <typename... _ArgTypes>
        MonoObject *operator()( _ArgTypes... aArgs )
        {
            void *lParameters[] = { (void *)&aArgs... };

            return mInstance->InvokeMethod( mMonoMethod, lParameters );
        }

      private:
        MonoMethod         *mMonoMethod;
        DotNetInstance *mInstance;
    };
} // namespace SE::Core