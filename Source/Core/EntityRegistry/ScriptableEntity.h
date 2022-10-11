/// @file   ScriptableEntity.h
///
/// @brief  Components that makes an entity scriptable using C++
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#pragma once

#include "Core/Math/Types.h"
#include "Core/Memory.h"
#include "Entity.h"

#include <filesystem>

namespace fs = std::filesystem;

namespace LTSE::Core::Internal
{

    template <typename ParentType> class BehaviourController
    {
      public:
        virtual ~BehaviourController() = default;

        template <typename T> T &Get() { return mEntity.Get<T>(); }
        template <typename T> bool Has() { return mEntity.Has<T>(); }

        virtual void Initialize( Entity<ParentType> aEntity ) { mEntity = aEntity; }

        virtual void OnCreate() {}
        virtual void OnDestroy() {}
        virtual void OnUpdate( Timestep ts ) {}

        Entity<ParentType> GetControlledEntity() const { return mEntity; };

      private:
        Entity<ParentType> mEntity;
    };

    template <typename ParentType> struct sBehaviourComponent
    {
        BehaviourController<ParentType> *ControllerInstance = nullptr;

        std::function<BehaviourController<ParentType> *()> InstantiateController;
        std::function<void( sBehaviourComponent<ParentType> * )> DestroyController;

        template <typename T, typename... Args> void Bind( Args &&...args )
        {
            InstantiateController = [&]() { return reinterpret_cast<BehaviourController<ParentType> *>( new T( std::forward<Args>( args )... ) ); };

            DestroyController = [&]( sBehaviourComponent<ParentType> *aNsc )
            {
                delete aNsc->ControllerInstance;
                aNsc->ControllerInstance = nullptr;
            };
        }
    };

} // namespace LTSE::Core::Internal
