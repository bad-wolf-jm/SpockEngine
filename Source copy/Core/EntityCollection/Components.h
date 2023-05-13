/// @file   Components.h.h
///
/// @brief  Basic components for entities
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#pragma once

#include <functional>
#include <optional>
#include <random>
#include <string>

#include "Core/Math/Types.h"

#include "Entity.h"
#include "Mono/MonoRuntime.h"
#include "Mono/MonoScriptMethod.h"

#ifdef LITTLEENDIAN
#    undef LITTLEENDIAN
#endif
#include <uuid_v4.h>

using namespace math;
namespace SE::Core
{
    struct sTag
    {
        std::string mValue;

        sTag()               = default;
        sTag( const sTag & ) = default;
        sTag( const std::string &aTag )
            : mValue( aTag )
        {
        }
    };

    struct sUUID
    {
        UUIDv4::UUID mValue;

        sUUID()
        {
            UUIDv4::UUIDGenerator<std::mt19937_64> lUuidGenerator;
            mValue = lUuidGenerator.getUUID();
        }
        sUUID( const sUUID & ) = default;
        sUUID( const std::string &aStringUUID )
            : mValue{ UUIDv4::UUID::fromStrFactory( aStringUUID ) }
        {
        }
        sUUID( const UUIDv4::UUID &aValue )
            : mValue{ aValue }
        {
        }
    };

    namespace Internal
    {
        template <typename ParentType>
        struct sRelationship
        {
            Internal::Entity<ParentType>              mParent{ entt::null, nullptr };
            std::vector<Internal::Entity<ParentType>> mChildren = {};

            sRelationship()                        = default;
            sRelationship( const sRelationship & ) = default;
        };

        template <typename ParentType, typename _Ty>
        struct sJoin
        {
            Internal::Entity<ParentType> mJoinEntity{}; //!< Handle to the joined entity

            /// @brief Retrieves a reference to the joined component
            _Ty &JoinedComponent() { return mJoinEntity.Get<_Ty>(); }

            sJoin()                = default;
            sJoin( const sJoin & ) = default;
        };

        template <typename ParentType>
        struct sBehaviourController
        {
          public:
            virtual ~sBehaviourController() = default;

            template <typename T>
            T &Get()
            {
                return mEntity.Get<T>();
            }
            template <typename T>
            bool Has()
            {
                return mEntity.Has<T>();
            }

            virtual void Initialize( Internal::Entity<ParentType> aEntity ) { mEntity = aEntity; }

            virtual void OnBeginScenario() {}
            virtual void OnEndScenario() {}
            virtual void OnTick( Timestep ts ) {}

            Internal::Entity<ParentType> GetControlledEntity() const { return mEntity; };

          private:
            Internal::Entity<ParentType> mEntity;
        };

        template <typename ParentType>
        struct sBehaviourComponent
        {
            sBehaviourController<ParentType> *ControllerInstance = nullptr;

            std::function<sBehaviourController<ParentType> *()>      InstantiateController;
            std::function<void( sBehaviourComponent<ParentType> * )> DestroyController;

            template <typename T, typename... Args>
            void Bind( Args &&...args )
            {
                InstantiateController = [&]()
                { return reinterpret_cast<sBehaviourController<ParentType> *>( new T( std::forward<Args>( args )... ) ); };

                DestroyController = [&]( sBehaviourComponent<ParentType> *aNsc )
                {
                    delete aNsc->ControllerInstance;
                    aNsc->ControllerInstance = nullptr;
                };
            }
        };

        template <typename ParentType>
        struct sMonoActor
        {
            std::string mClassFullName = "";

            MonoScriptClass         mClass;
            Ref<MonoScriptInstance> mInstance;
            Ref<MonoScriptInstance> mEntityInstance;

            sMonoActor()                     = default;
            sMonoActor( const sMonoActor & ) = default;

            ~sMonoActor() = default;

            sMonoActor( const std::string &aClassFullName )
                : mClassFullName{ aClassFullName }

            {
                size_t      lSeparatorPos   = aClassFullName.find_last_of( '.' );
                std::string lClassNamespace = aClassFullName.substr( 0, lSeparatorPos );
                std::string lClassName      = aClassFullName.substr( lSeparatorPos + 1 );

                mClass = MonoRuntime::GetClassType( mClassFullName );
            }

            template <typename T>
            T &Get()
            {
                return mEntity.Get<T>();
            }

            void Initialize( Internal::Entity<ParentType> aEntity )
            {
                mEntity = aEntity;

                // Create Mono side entity object
                auto lEntityID    = static_cast<uint32_t>( mEntity );
                auto lRegistryID  = (size_t)mEntity.GetRegistry();
                auto lEntityClass = MonoRuntime::GetClassType( "SpockEngine.Entity" );
                mEntityInstance   = lEntityClass.Instantiate( &lEntityID, &lRegistryID );

                if( mClassFullName.empty() ) return;

                size_t      lSeparatorPos   = mClassFullName.find_last_of( '.' );
                std::string lClassNamespace = mClassFullName.substr( 0, lSeparatorPos );
                std::string lClassName      = mClassFullName.substr( lSeparatorPos + 1 );

                mClass = MonoRuntime::GetClassType( mClassFullName );

                // Instantiate the Mono actor class with the entity object as parameter
                mInstance = mClass.Instantiate();
                mInstance->CallMethod( "Initialize", (size_t)mEntityInstance->GetInstance() );
            }

            void OnBeginScenario() { mInstance->InvokeMethod( "BeginScenario", 0, nullptr ); }

            void OnEndScenario() { mInstance->InvokeMethod( "EndScenario", 0, nullptr ); }

            void OnTick( Timestep ts ) { mInstance->CallMethod( "Tick", ts.GetMilliseconds() ); }

            Internal::Entity<ParentType> GetControlledEntity() const { return mEntity; };

          private:
            Internal::Entity<ParentType> mEntity;

            MonoScriptMehod mOnUpdate{};
        };

        template <typename ParentType>
        struct sMonoUIComponent
        {
            float mX               = 0.0f;
            float mY               = 0.0f;
            float mWidth           = 100.0f;
            float mHeight          = 100.0f;
            vec4  mFillColor       = { 0.0f, 0.0f, 0.0f, 0.5f };
            vec4  mBorderColor     = { 1.0f, 1.0f, 1.0f, 0.75f };
            float mBorderThickness = 2.0f;
            float mRounding        = 5.0f;

            bool mDisplayInEditor = true;
            bool mPreview         = true;

            std::string             mClassFullName = "";
            MonoScriptClass         mClass;
            Ref<MonoScriptInstance> mInstance;
            Ref<MonoScriptInstance> mPreviewInstance;
            Ref<MonoScriptInstance> mEntityInstance;

            sMonoUIComponent()                           = default;
            sMonoUIComponent( const sMonoUIComponent & ) = default;

            ~sMonoUIComponent() = default;

            sMonoUIComponent( const std::string &aClassFullName )
                : mClassFullName{ aClassFullName }

            {
            }

            template <typename T>
            T &Get()
            {
                return mEntity.Get<T>();
            }

            void Initialize( Internal::Entity<ParentType> aEntity )
            {
                mEntity           = aEntity;
                auto lEntityID    = static_cast<uint32_t>( mEntity );
                auto lRegistryID  = (size_t)mEntity.GetRegistry();
                auto lEntityClass = MonoRuntime::GetClassType( "SpockEngine.Entity" );
                mEntityInstance   = lEntityClass.Instantiate( &lEntityID, &lRegistryID );

                if( mClassFullName.empty() ) return;

                size_t      lSeparatorPos   = mClassFullName.find_last_of( '.' );
                std::string lClassNamespace = mClassFullName.substr( 0, lSeparatorPos );
                std::string lClassName      = mClassFullName.substr( lSeparatorPos + 1 );

                mClass = MonoRuntime::GetClassType( mClassFullName );

                if( mDisplayInEditor && mPreview )
                {
                    mPreviewInstance = mClass.Instantiate();
                    mPreviewInstance->CallMethod( "Initialize", (size_t)mEntityInstance->GetInstance() );
                }
            }

            void OnBeginScenario()
            {
                // Instantiate the Mono actor class with the entity object as parameter
                if( mClassFullName.empty() ) return;

                mInstance = mClass.Instantiate();

                mInstance->CallMethod( "Initialize", (size_t)mEntityInstance->GetInstance() );
                mInstance->InvokeMethod( "BeginScenario", 0, nullptr );
            }

            void OnEndScenario()
            {
                mInstance->InvokeMethod( "EndScenario", 0, nullptr );
                mInstance = MonoScriptInstance{};
            }

            void OnUpdate( Timestep ts )
            {
                if( mInstance ) mInstance->CallMethod( "DrawContent", ts.GetMilliseconds() );
            }

            void OnPreviewUpdate( Timestep ts )
            {
                if( mPreviewInstance ) mPreviewInstance->CallMethod( "DrawPreviewContent", ts.GetMilliseconds() );
            }

            Internal::Entity<ParentType> GetControlledEntity() const { return mEntity; };

          private:
            Internal::Entity<ParentType> mEntity;

            MonoScriptMehod mOnUpdate{};
        };

    } // namespace Internal
} // namespace SE::Core
