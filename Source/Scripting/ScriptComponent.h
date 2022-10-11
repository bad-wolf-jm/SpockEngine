#pragma once

#include "ScriptingEngine.h"

#include "Core/EntityRegistry/Registry.h"
#include "Core/Memory.h"

namespace LTSE::Core
{
    class sLuaScriptComponent
    {
      public:
        sLuaScriptComponent( Ref<ScriptingEngine> aScriptingEngine, fs::path aScriptFile )
            : mScriptFile{ aScriptFile }
            , mScriptingEngine{ aScriptingEngine }
        {
            //
        }

        ~sLuaScriptComponent() = default;

        sLuaScriptComponent( sLuaScriptComponent const & ) = default;

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
        Ref<ScriptingEngine> mScriptingEngine = nullptr;
        ScriptEnvironment mScriptEnvironment;
        fs::path mScriptFile;
        Entity mEntity;
    };
} // namespace LTSE::Core