#include "InternalCalls.h"
#include "EntityRegistry.h"
#include "TypeReflection.h"
#include <iostream>
#include <list>
#include <string>

#include "Core/Logging.h"

#include "Runtime.h"
#include "Utils.h"

namespace SE::Core::Interop
{
    extern "C"
    {
        void Engine_Initialize( CLRVec2 aPosition, CLRVec2 aSize, sUIConfiguration aUIConfiguration )
        {
            math::ivec2 lWindowSize     = { aSize.x, aSize.y };
            math::ivec2 lWindowPosition = { aPosition.x, aPosition.y };

            UIConfiguration lUIConfig{};
            lUIConfig.mFontSize       = (int)aUIConfiguration.mFontSize;
            lUIConfig.mMainFont       = ConvertStringForCoreclr( aUIConfiguration.mMainFont );
            lUIConfig.mBoldFont       = ConvertStringForCoreclr( aUIConfiguration.mBoldFont );
            lUIConfig.mItalicFont     = ConvertStringForCoreclr( aUIConfiguration.mItalicFont );
            lUIConfig.mBoldItalicFont = ConvertStringForCoreclr( aUIConfiguration.mBoldItalicFont );
            lUIConfig.mMonoFont       = ConvertStringForCoreclr( aUIConfiguration.mMonoFont );
            lUIConfig.mIconFont       = ConvertStringForCoreclr( aUIConfiguration.mIconFont );

            SE::Core::Engine::Initialize( lWindowSize, lWindowPosition, ConvertStringForCoreclr( aUIConfiguration.mIniFile ),
                                          lUIConfig );
        }

        void Engine_Main( UpdateFn aUpdateDelegate, RenderSceneFn aRenderDelegate, RenderUIFn aRenderUIDelegate, RenderMenuFn aRenderMenuDelegate )
        {
            SE::OtdrEditor::Application lEditorApplication( aUpdateDelegate, aRenderDelegate, aRenderUIDelegate, aRenderMenuDelegate );

            SE::Core::Engine::GetInstance()->UpdateDelegate.connect<&SE::OtdrEditor::Application::Update>( lEditorApplication );
            SE::Core::Engine::GetInstance()->RenderDelegate.connect<&SE::OtdrEditor::Application::RenderScene>( lEditorApplication );
            SE::Core::Engine::GetInstance()->UIDelegate.connect<&SE::OtdrEditor::Application::RenderUI>( lEditorApplication );

            while( SE::Core::Engine::GetInstance()->Tick() )
            {
            }
        }

        void Engine_Shutdown() { SE::Core::Engine::Shutdown(); }
    }

    // uint32_t Entity_Create( EntityCollection *aRegistry, MonoString *aName, uint32_t aEntityID )
    // {
    //     auto lName      = string_t( mono_string_to_utf8( aName ) );
    //     auto lNewEntity = aRegistry->CreateEntity( aRegistry->WrapEntity( static_cast<entt::entity>( aEntityID ) ), lName );

    //     return static_cast<uint32_t>( lNewEntity );
    // }

    // bool Entity_IsValid( uint32_t aEntityID, EntityCollection *aRegistry )
    // {
    //     return aRegistry->WrapEntity( static_cast<entt::entity>( aEntityID ) ).IsValid();
    // }

    // bool Entity_Has( uint32_t aEntityID, EntityCollection *aRegistry, MonoReflectionType *aComponentType )
    // {
    //     MonoType *lMonoType = mono_reflection_type_get_type( aComponentType );

    //     const entt::meta_type lMetaType = Core::GetMetaType( lMonoType );
    //     const entt::meta_any  lMaybeAny =
    //         Core::InvokeMetaFunction( lMetaType, "Has"_hs, aRegistry->WrapEntity( static_cast<entt::entity>( aEntityID ) ) );

    //     return lMaybeAny.cast<bool>();
    // }

    // MonoObject *Entity_Get( uint32_t aEntityID, EntityCollection *aRegistry, MonoReflectionType *aComponentType )
    // {
    //     MonoType *lMonoType = mono_reflection_type_get_type( aComponentType );

    //     const entt::meta_type lMetaType = Core::GetMetaType( lMonoType );
    //     const entt::meta_any  lMaybeAny = Core::InvokeMetaFunction(
    //         lMetaType, "Get"_hs, aRegistry->WrapEntity( static_cast<entt::entity>( aEntityID ) ), DotNetClass( lMonoType ) );

    //     return lMaybeAny.cast<DotNetInstance>().GetInstance();
    // }

    // void Entity_Replace( uint32_t aEntityID, EntityCollection *aRegistry, MonoReflectionType *aComponentType,
    //                      MonoObject *aNewComponent )
    // {
    //     MonoType  *lMonoType  = mono_reflection_type_get_type( aComponentType );
    //     MonoClass *lMonoClass = mono_class_from_mono_type( lMonoType );

    //     const entt::meta_type lMetaType = Core::GetMetaType( lMonoType );
    //     Core::InvokeMetaFunction( lMetaType, "Replace"_hs, aRegistry->WrapEntity( static_cast<entt::entity>( aEntityID ) ),
    //                               DotNetInstance( lMonoClass, aNewComponent ) );
    // }

    // void Entity_Add( uint32_t aEntityID, EntityCollection *aRegistry, MonoReflectionType *aComponentType, MonoObject *aNewComponent )
    // {
    //     MonoType  *lMonoType  = mono_reflection_type_get_type( aComponentType );
    //     MonoClass *lMonoClass = mono_class_from_mono_type( lMonoType );

    //     const entt::meta_type lMetaType = Core::GetMetaType( lMonoType );
    //     Core::InvokeMetaFunction( lMetaType, "Add"_hs, aRegistry->WrapEntity( static_cast<entt::entity>( aEntityID ) ),
    //                               DotNetInstance( lMonoClass, aNewComponent ) );
    // }

    // void Entity_Remove( uint32_t aEntityID, EntityCollection *aRegistry, MonoReflectionType *aComponentType )
    // {
    //     MonoType *lMonoType = mono_reflection_type_get_type( aComponentType );

    //     const entt::meta_type lMetaType = Core::GetMetaType( lMonoType );
    //     const entt::meta_any  lMaybeAny =
    //         Core::InvokeMetaFunction( lMetaType, "Remove"_hs, aRegistry->WrapEntity( static_cast<entt::entity>( aEntityID ) ) );
    // }

} // namespace SE::Core::Interop