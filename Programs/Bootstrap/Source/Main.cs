using System;
using System.IO;
using System.Linq;
using System.Drawing;
using System.Drawing.Text;
using System.Windows.Media;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Reflection;

using CommandLine;
using SpockEngine;
using Math = SpockEngine.Math;

using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

class SEBootstrap
{

    static Assembly mApplication;
    static Type mApplicationStaticClass;

    static MethodInfo mUpdateMenu;
    static MethodInfo mUpdate;
    static MethodInfo mUpdateUI;

    static void UpdateDelegate(float aTs)
    {
        mUpdate?.Invoke(null, new object[] { aTs });
    }

    static bool RenderUIDelegate(float aTs)
    {
        object lShouldQuit = mUpdateUI?.Invoke(null, new object[] { aTs });

        return (lShouldQuit != null) ? (bool)lShouldQuit : false;
    }

    static void RenderDelegate() {}

    static bool RenderMenuDelegate()
    {
        object lShouldQuit = mUpdateMenu?.Invoke(null, new object[] { });

        return (lShouldQuit != null) ? (bool)lShouldQuit : false;
    }

    delegate void UpdateDelegateType(float aTs);
    delegate void RenderDelegateType();
    delegate bool RenderUIDelegateType(float aTs);
    delegate bool RenderMenuDelegateType();

    static void RunEngineMainFunction()
    {
        UpdateDelegateType lUpdateDelegatePtr = UpdateDelegate;
        RenderDelegateType lRenderDelegateTypePtr = RenderDelegate;
        RenderUIDelegateType lRenderUIDelegateTypePtr = RenderUIDelegate;
        RenderMenuDelegateType lRenderMenuDelegateTypePtr = RenderMenuDelegate;

        CppCall.Engine_Main(
            Marshal.GetFunctionPointerForDelegate(lUpdateDelegatePtr),
            Marshal.GetFunctionPointerForDelegate(lRenderDelegateTypePtr),
            Marshal.GetFunctionPointerForDelegate(lRenderUIDelegateTypePtr),
            Marshal.GetFunctionPointerForDelegate(lRenderMenuDelegateTypePtr));
    }

    static void GuardedMain(Options aOpt)
    {
        var lExePath = Directory.GetParent(System.Reflection.Assembly.GetExecutingAssembly().Location).FullName;
        var lUserHome = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        var lUserConfig = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);

        var lUIConfiguration = new UIConfiguration();
        lUIConfiguration.mFontSize = 15.0f;
        lUIConfiguration.mMainFont = Path.Combine(new string[] { lExePath, "Fonts", "Roboto-Regular.ttf" });
        lUIConfiguration.mBoldFont = Path.Combine(new string[] { lExePath, "Fonts", "Roboto-Bold.ttf" });
        lUIConfiguration.mItalicFont = Path.Combine(new string[] { lExePath, "Fonts", "Roboto-Italic.ttf" });
        lUIConfiguration.mBoldItalicFont = Path.Combine(new string[] { lExePath, "Fonts", "Roboto-BoldItalic.ttf" });
        lUIConfiguration.mIconFont = Path.Combine(new string[] { lExePath, "Fonts", "fontawesome-webfont.ttf" });
        lUIConfiguration.mMonoFont = Path.Combine(new string[] { lExePath, "Fonts", "DejaVuSansMono.ttf" });

        var lAppConfigRoot = Path.Combine(lUserConfig, aOpt.Application);

        var lAppConfig = Utilities.CreateFolder(new string[] { lAppConfigRoot, "Config" });
        var lAppLogs = Utilities.CreateFolder(new string[] { lAppConfigRoot, "Logs" });

        lUIConfiguration.mIniFile = Path.Combine(lAppConfig, "imgui.ini");
        var lAppIniConfig = Path.Combine(lAppConfig, "Application.yaml");

        float lX = 0.0f;
        float lY = 0.0f;
        float lWidth = 1920.0f;
        float lHeight = 1000.0f;
        CppCall.Engine_Initialize(new Math.vec2(lX, lY), new Math.vec2(lWidth, lHeight), lUIConfiguration);

        // Load application assembly
        var lApplicationPath = Path.Combine(new string[] { @"D:\Build\Lib", "Debug", "develop", "net462", aOpt.Application, $"{aOpt.Application}.dll" });
        mApplication = Assembly.LoadFrom(lApplicationPath);
        mApplicationStaticClass = mApplication.GetType($"{aOpt.Application}.{aOpt.Application}");

        mUpdateMenu = mApplicationStaticClass.GetMethod("UpdateMenu");
        mUpdate = mApplicationStaticClass.GetMethod("Update");
        mUpdateUI = mApplicationStaticClass.GetMethod("UpdateUI");

        mApplicationStaticClass.GetMethod("Configure").Invoke(null, new object[] { lAppIniConfig });

        RunEngineMainFunction();

        mApplicationStaticClass.GetMethod("Teardown").Invoke(null, new object[] { lAppIniConfig });

        CppCall.Engine_Shutdown();
    }

    static void Main(string[] args)
    {
        Parser.Default.ParseArguments<Options>(args).WithParsed<Options>(GuardedMain);
    }
}