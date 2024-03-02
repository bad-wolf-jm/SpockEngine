#pragma once

#include "ScriptingEngine.h"

#include "Core/Memory.h"
#include "Core/Entity/Collection.h"

namespace SE::Core
{
    class sLuaScriptComponent
    {
      public:
        sLuaScriptComponent( ref_t<ScriptingEngine> aScriptingEngine, fs::path aScriptFile )
            : mScriptFile{ aScriptFile }
            , mScriptingEngine{ aScriptingEngine }
        {
            //
        }

        ~sLuaScriptComponent() = default;

        template <typename T> T &Get() { return mEntity.Get<T>(); }

        virtual void Initialize( Entity aEntity ) { mEntity = aEntity; }

        virtual void OnCreate()
        {
            mScriptEnvironment = mScriptingEngine->LoadFile( mScriptFile.string() );
            mScriptEnvironment["Initialize"]();
        }

        virtual void OnDestroy() { mScriptEnvironment["Shutdown"](); }

        virtual void OnUpdate( Timestep ts ) { mScriptEnvironment["Update"]( static_cast<float>( ts ) ); }

        Entity GetControlledEntity() const { return mEntity; };

      private:
        ref_t<ScriptingEngine> mScriptingEngine = nullptr;
        ScriptEnvironment mScriptEnvironment;
        fs::path mScriptFile;
        Entity mEntity;
    };
} // namespace SE::Core