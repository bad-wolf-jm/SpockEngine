#include "EditComponent.h"

#include "UI/UI.h"
#include "UI/Widgets.h"

namespace SE::Core
{
    bool EditComponent( sActorComponent &aComponent )
    {
        UI::Text( "Component class:" );
        UI::Text( "{}", aComponent.mClassFullName );

        for( const auto &[name, field] : aComponent.mClass.GetFields() )
        {
            UI::Text( "{}", name );
        }

        return false;
    }

    bool EditComponent( sUIComponent &aComponent )
    {
        UI::Text( "Component class:" );
        UI::Text( "{}", aComponent.mClassFullName );

        for( const auto &[name, field] : aComponent.mClass.GetFields() )
        {
            UI::Text( "{}", name );
        }

        return false;
    }

} // namespace SE::Core