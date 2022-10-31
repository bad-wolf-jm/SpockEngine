

int main( int argc, char **argv )
{
    // EchoDSMVPEditor g_EditorWindow{};

    // g_EditorWindow.Init();

    ScriptManager::Initialize();

    while( mEngineLoop->Tick() )
    {
    }

    SaveConfiguration();

    mEngineLoop->Shutdown();
    ScriptManager::Shutdown();

    return 0;
}
